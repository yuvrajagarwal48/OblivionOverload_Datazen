/**
 * Cytoscape.js stylesheet — visual encoding per UI_SPECS §7
 *
 * Nodes:
 *   tier     → size (tier 1 = largest)
 *   capital_ratio → color (green 0.15+ → yellow 0.08 → red 0.0)
 *   stress   → border glow intensity
 *   status=defaulted → greyed / faded
 *
 * Edges:
 *   weight   → thickness
 *   directed → arrow
 *   type     → solid style
 */

function capitalRatioToColor(ratio) {
  // Green (healthy) → Yellow (stressed) → Red (critical) — adjusted for white background
  const r = ratio ?? 0.1;
  if (r >= 0.15) return '#10b981';  // emerald green
  if (r >= 0.12) return '#22c55e';  // lighter green
  if (r >= 0.10) return '#84cc16';  // lime green
  if (r >= 0.08) return '#f59e0b';  // amber
  if (r >= 0.05) return '#f97316';  // orange
  if (r >= 0.03) return '#ef4444';  // red
  return '#dc2626';                  // darker red
}

function tierToSize(tier) {
  switch (tier) {
    case 1: return 55;
    case 2: return 40;
    case 3: return 28;
    default: return 35;
  }
}

function stressToBorderWidth(stress) {
  const s = stress ?? 0;
  return 2 + s * 6; // 2px to 8px
}

function stressToBorderColor(stress) {
  const s = stress ?? 0;
  if (s < 0.3) return 'rgba(16, 185, 129, 0.5)';   // emerald
  if (s < 0.6) return 'rgba(245, 158, 11, 0.7)';  // amber
  return 'rgba(239, 68, 68, 0.9)';                 // red
}

function weightToWidth(weight) {
  const w = weight ?? 10;
  return Math.max(1, Math.min(w / 10, 8));
}

function edgeTypeColor(activityType) {
  switch (activityType) {
    case 'LENDING': return '#3b82f6';    // blue — money flowing
    case 'REPAYMENT': return '#10b981';  // green — repayment
    case 'NEW_LINK': return '#8b5cf6';   // purple — new relationship
    default: return '#cbd5e1';           // grey — static
  }
}

/**
 * Returns the Cytoscape stylesheet array.
 * Data-driven mappers use ele.data() accessors.
 */
export function getCytoscapeStylesheet() {
  return [
    // ─── NODE BASE ───
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': '11px',
        'font-weight': '600',
        'font-family': "'Inter', 'Segoe UI', sans-serif",
        'color': '#ffffff',
        'text-outline-color': '#1e293b',
        'text-outline-width': 2.5,
        'width': (ele) => tierToSize(ele.data('tier')),
        'height': (ele) => tierToSize(ele.data('tier')),
        'background-color': (ele) => capitalRatioToColor(ele.data('capital_ratio')),
        'border-width': (ele) => stressToBorderWidth(ele.data('stress')),
        'border-color': (ele) => stressToBorderColor(ele.data('stress')),
        'border-opacity': 0.8,
        'overlay-opacity': 0,
        'transition-property': 'background-color, border-color, border-width, width, height, opacity',
        'transition-duration': '300ms',
        'transition-timing-function': 'ease-in-out',
      },
    },

    // ─── NODE WITH DECISION LABEL ───
    {
      selector: 'node[last_decision]',
      style: {
        'label': (ele) => {
          const decision = ele.data('last_decision');
          if (!decision) return ele.data('label');
          const shortDecision = decision.replace(/_/g, ' ');
          return `${ele.data('label')}\n${shortDecision}`;
        },
        'text-wrap': 'wrap',
        'text-max-width': '80px',
        'font-size': '9px',
        'text-valign': 'bottom',
        'text-margin-y': 8,
      },
    },

    // ─── DEFAULTED NODE ───
    {
      selector: 'node[status = "defaulted"]',
      style: {
        'background-color': '#94a3b8',
        'border-color': '#cbd5e1',
        'border-width': 2,
        'opacity': 0.4,
        'color': '#64748b',
      },
    },

    // ─── SELECTED NODE ───
    {
      selector: 'node:selected',
      style: {
        'border-color': '#3b82f6',
        'border-width': 4,
        'overlay-color': '#3b82f6',
        'overlay-opacity': 0.15,
      },
    },

    // ─── EDGE BASE ───
    {
      selector: 'edge',
      style: {
        'width': (ele) => weightToWidth(ele.data('weight')),
        'line-color': (ele) => edgeTypeColor(ele.data('activity_type')),
        'target-arrow-color': (ele) => edgeTypeColor(ele.data('activity_type')),
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.8,
        'curve-style': 'bezier',
        'opacity': 0.5,
        'label': (ele) => ele.data('activity_label') || '',
        'font-size': '8px',
        'font-weight': '600',
        'color': '#e2e8f0',
        'text-background-color': '#1e293b',
        'text-background-opacity': 0.85,
        'text-background-padding': '3px',
        'text-background-shape': 'roundrectangle',
        'text-rotation': 'autorotate',
        'text-margin-y': -8,
        'edge-text-rotation': 'autorotate',
        'transition-property': 'line-color, opacity, width',
        'transition-duration': '300ms',
      },
    },

    // ─── ACTIVE EDGE (animated) ───
    {
      selector: 'edge.animated',
      style: {
        'line-color': '#3b82f6',
        'target-arrow-color': '#3b82f6',
        'opacity': 1,
      },
    },

    // ─── FADING EDGE (default contagion) ───
    {
      selector: 'edge.fading',
      style: {
        'line-color': '#dc2626',
        'target-arrow-color': '#dc2626',
        'opacity': 0.25,
      },
    },

    // ─── EVENT ANIMATION CLASSES ───
    {
      selector: 'node.hoard-effect',
      style: {
        'border-color': '#f59e0b',
        'border-width': 8,
        'width': (ele) => tierToSize(ele.data('tier')) * 0.75,
        'height': (ele) => tierToSize(ele.data('tier')) * 0.75,
      },
    },
    {
      selector: 'node.fire-sale-effect',
      style: {
        'border-color': '#ef4444',
        'border-width': 10,
        'border-opacity': 0.7,
      },
    },
    {
      selector: 'node.default-effect',
      style: {
        'background-color': '#94a3b8',
        'border-color': '#cbd5e1',
        'width': 15,
        'height': 15,
        'opacity': 0.3,
      },
    },
    {
      selector: 'node.price-drop-effect',
      style: {
        'opacity': 0.5,
      },
    },
    {
      selector: 'edge.price-drop-effect',
      style: {
        'opacity': 0.15,
      },
    },
  ];
}

// Export helpers for external use
export { capitalRatioToColor, tierToSize };
