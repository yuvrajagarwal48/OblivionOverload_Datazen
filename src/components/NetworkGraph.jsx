import React, { useRef, useEffect, useCallback } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { Network } from 'lucide-react';
import useSimulationStore from '../store/simulationStore';
import { getCytoscapeStylesheet } from '../systems/cytoscapeStyles';
import { dispatchAnimation } from '../systems/eventAnimations';
import './NetworkGraph.css';

/**
 * Network Graph — the centerpiece.
 *
 * Uses Cytoscape.js via react-cytoscapejs for initial mount.
 * All subsequent data updates are done imperatively via cy ref.
 *
 * Rules (from spec §11):
 *   - Layout computed ONCE on first STATE_UPDATE
 *   - On subsequent STATE_UPDATEs: batch-update data, do NOT relayout
 *   - On EVENTs: dispatch animation, auto-remove
 *   - Click-to-select nodes → bank view
 */
export default function NetworkGraph() {
  const cyRef = useRef(null);
  const prevEventsLenRef = useRef(0);

  const nodes = useSimulationStore((s) => s.nodes);
  const edges = useSimulationStore((s) => s.edges);
  const events = useSimulationStore((s) => s.events);
  const nodeDecisions = useSimulationStore((s) => s.nodeDecisions || {});
  const edgeActivity = useSimulationStore((s) => s.edgeActivity || {});
  const layoutComputed = useSimulationStore((s) => s.layoutComputed);
  const setLayoutComputed = useSimulationStore((s) => s.setLayoutComputed);
  const toggleBankSelection = useSimulationStore((s) => s.toggleBankSelection);
  const simStatus = useSimulationStore((s) => s.simStatus);

  // Convert backend nodes/edges to Cytoscape elements format
  const toCyElements = useCallback((nodeList, edgeList, decisions, activity) => {
    const cyNodes = nodeList.map((n) => ({
      group: 'nodes',
      data: {
        id: String(n.id),
        label: `B${n.id}`,
        tier: n.tier,
        capital_ratio: n.capital_ratio,
        stress: n.stress,
        status: n.status,
        last_updated_timestep: n.last_updated_timestep,
        last_decision: decisions[n.id] || decisions[String(n.id)] || '',
      },
    }));

    const cyEdges = edgeList.map((e, i) => {
      const key = `${e.source}-${e.target}`;
      const act = activity[key];
      return {
        group: 'edges',
        data: {
          id: `e-${e.source}-${e.target}-${i}`,
          source: String(e.source),
          target: String(e.target),
          weight: e.weight,
          type: e.type || 'credit',
          activity_label: act?.label || '',
          activity_type: act?.type || '',
        },
      };
    });

    return [...cyNodes, ...cyEdges];
  }, []);

  // Handle data updates — layout once, then batch-update only
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || nodes.length === 0) return;

    if (!layoutComputed) {
      // FIRST STATE_UPDATE: set elements and run layout ONCE
      cy.json({ elements: toCyElements(nodes, edges, nodeDecisions, edgeActivity) });

      const layout = cy.layout({
        name: 'cose',
        animate: true,
        animationDuration: 800,
        fit: true,
        padding: 40,
        nodeRepulsion: () => 8000,
        idealEdgeLength: () => 100,
        edgeElasticity: () => 100,
        gravity: 0.25,
        randomize: true,
      });

      layout.run();
      setLayoutComputed(true);
    } else {
      // SUBSEQUENT STATE_UPDATEs: batch-update data only, NO relayout
      cy.batch(() => {
        nodes.forEach((n) => {
          const cyNode = cy.getElementById(String(n.id));
          if (cyNode.length) {
            cyNode.data({
              tier: n.tier,
              capital_ratio: n.capital_ratio,
              stress: n.stress,
              status: n.status,
              last_updated_timestep: n.last_updated_timestep,
              last_decision: nodeDecisions[n.id] || nodeDecisions[String(n.id)] || '',
            });
          }
        });

        edges.forEach((e, i) => {
          const edgeId = `e-${e.source}-${e.target}-${i}`;
          const cyEdge = cy.getElementById(edgeId);
          const key = `${e.source}-${e.target}`;
          const act = edgeActivity[key];
          if (cyEdge.length) {
            cyEdge.data({
              weight: e.weight,
              type: e.type || 'credit',
              activity_label: act?.label || '',
              activity_type: act?.type || '',
            });
          }
        });
      });
    }
  }, [nodes, edges, nodeDecisions, edgeActivity, layoutComputed, setLayoutComputed, toCyElements]);

  // Handle events — dispatch animations for new events
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !events.length) return;

    // Only process newly added events
    const newEvents = events.slice(prevEventsLenRef.current);
    prevEventsLenRef.current = events.length;

    newEvents.forEach((event) => {
      dispatchAnimation(cy, event);
    });
  }, [events]);

  // Set up click handler for bank selection
  const handleCyInit = useCallback(
    (cy) => {
      cyRef.current = cy;

      cy.on('tap', 'node', (evt) => {
        const nodeId = evt.target.id();
        toggleBankSelection(nodeId);
      });

      // Dark background
      cy.style().selector('core').style({
        'active-bg-color': '#58a6ff',
        'active-bg-opacity': 0.1,
      });
    },
    [toggleBankSelection]
  );

  // Reset cy state when simulation resets
  useEffect(() => {
    if (simStatus === 'idle' && cyRef.current) {
      cyRef.current.elements().remove();
      prevEventsLenRef.current = 0;
    }
  }, [simStatus]);

  return (
    <div className="network-graph-container">
      {nodes.length === 0 ? (
        <div className="graph-placeholder">
          <div className="placeholder-icon"><Network size={48} strokeWidth={1.5} /></div>
          <div className="placeholder-text">Select a scenario and run the simulation</div>
          <div className="placeholder-sub">The network graph will appear here</div>
        </div>
      ) : null}
      <CytoscapeComponent
        elements={[]}
        stylesheet={getCytoscapeStylesheet()}
        layout={{ name: 'preset' }}
        cy={handleCyInit}
        className="cytoscape-graph"
        style={{
          width: '100%',
          height: '100%',
          opacity: nodes.length === 0 ? 0 : 1,
        }}
        boxSelectionEnabled={false}
        autounselectify={false}
        userZoomingEnabled={true}
        userPanningEnabled={true}
        minZoom={0.3}
        maxZoom={3}
      />
    </div>
  );
}
