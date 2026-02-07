import { ANIMATION } from '../config';

/**
 * Event Animation System — visual explanations per UI_SPECS §8
 *
 * Each event type triggers ONE animation that auto-removes.
 * Animations never block rendering. State always overrides.
 *
 * Event types:
 *   LEND       → moving pulse along edge (source→target)
 *   HOARD      → node contraction + lock ring
 *   FIRE_SALE  → expanding ripple from node
 *   PRICE_DROP → subtle global dim
 *   DEFAULT    → node collapse + connected edge fade
 */

/**
 * Dispatch an animation for a single event.
 * @param {cytoscape.Core} cy  - Cytoscape instance
 * @param {Object} event       - Event payload from backend
 */
export function dispatchAnimation(cy, event) {
  if (!cy || cy.destroyed()) return;

  switch (event.event_type) {
    case 'LEND':
      animateLend(cy, event);
      break;
    case 'HOARD':
      animateHoard(cy, event);
      break;
    case 'FIRE_SALE':
      animateFireSale(cy, event);
      break;
    case 'PRICE_DROP':
      animatePriceDrop(cy, event);
      break;
    case 'DEFAULT':
      animateDefault(cy, event);
      break;
    default:
      console.warn('[Animation] Unknown event type:', event.event_type);
  }
}

/**
 * LEND — pulse along the edge from source to target
 */
function animateLend(cy, event) {
  const sourceId = String(event.from);
  const targetId = String(event.to);

  // Find the edge between source and target
  const edges = cy.edges(`[source = "${sourceId}"][target = "${targetId}"]`);
  if (edges.empty()) return;

  const edge = edges[0];
  edge.addClass('animated');

  setTimeout(() => {
    if (!cy.destroyed()) {
      edge.removeClass('animated');
    }
  }, ANIMATION.LEND_DURATION);
}

/**
 * HOARD — node contraction with lock ring effect
 */
function animateHoard(cy, event) {
  const nodeId = String(event.from);
  const node = cy.getElementById(nodeId);
  if (node.empty()) return;

  node.addClass('hoard-effect');

  setTimeout(() => {
    if (!cy.destroyed()) {
      node.removeClass('hoard-effect');
    }
  }, ANIMATION.HOARD_DURATION);
}

/**
 * FIRE_SALE — expanding ripple from the node
 */
function animateFireSale(cy, event) {
  const nodeId = String(event.from);
  const node = cy.getElementById(nodeId);
  if (node.empty()) return;

  node.addClass('fire-sale-effect');

  // Also briefly highlight connected edges
  const connectedEdges = node.connectedEdges();
  connectedEdges.addClass('animated');

  setTimeout(() => {
    if (!cy.destroyed()) {
      node.removeClass('fire-sale-effect');
      connectedEdges.removeClass('animated');
    }
  }, ANIMATION.FIRE_SALE_DURATION);
}

/**
 * PRICE_DROP — subtle global dim effect
 */
function animatePriceDrop(cy, event) {
  const allNodes = cy.nodes();
  const allEdges = cy.edges();

  allNodes.addClass('price-drop-effect');
  allEdges.addClass('price-drop-effect');

  setTimeout(() => {
    if (!cy.destroyed()) {
      allNodes.removeClass('price-drop-effect');
      allEdges.removeClass('price-drop-effect');
    }
  }, ANIMATION.PRICE_DROP_DURATION);
}

/**
 * DEFAULT — node collapse + connected edge fade
 */
function animateDefault(cy, event) {
  const nodeId = String(event.from);
  const node = cy.getElementById(nodeId);
  if (node.empty()) return;

  node.addClass('default-effect');

  // Fade connected edges
  const connectedEdges = node.connectedEdges();
  connectedEdges.addClass('fading');

  setTimeout(() => {
    if (!cy.destroyed()) {
      node.removeClass('default-effect');
      connectedEdges.removeClass('fading');
      // State update will set the authoritative visual afterward
    }
  }, ANIMATION.DEFAULT_DURATION);
}

/**
 * Process a batch of events — call dispatchAnimation for each.
 * Respects the ≤10 events/sec constraint by not throttling
 * (backend controls emission rate).
 */
export function processEventBatch(cy, events) {
  if (!cy || cy.destroyed() || !events?.length) return;
  events.forEach((event) => dispatchAnimation(cy, event));
}
