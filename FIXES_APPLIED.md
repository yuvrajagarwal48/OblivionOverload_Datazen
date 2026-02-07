# Recent Fixes Applied

## 1. Edge Label Updates (FIXED ✅)

**Problem:** Edge labels showed at start but vanished, then when fixed to persist, they stopped updating.

**Solution:** Modified `ingestEdges()` in `simulationStore.js`:
- Always show current weight on every edge: `₹${weight.toFixed(0)}`
- For edges with significant changes (|delta| > 0.5): show full message with delta and total
- For static edges: show just current weight
- Fresh `edgeActivity` state on each update (no stale data)

**Result:** Edge labels now persistently show current exposure amounts and update on every transaction.

## 2. CCP Node Support (ADDED ✅)

**Features Added:**
- `ingestTopologyNodes()` function to merge CCP nodes from topology endpoint
- CCP-specific styling in `cytoscapeStyles.js`:
  - Diamond shape
  - Purple color (#8b5cf6)
  - 60x60px size
  - "CCP-X" label format
- Updated `NetworkGraph` to pass `node_type` field
- `useSimulationControl` now calls `ingestTopologyNodes()` after fetching topology

**Result:** CCP nodes will appear as purple diamonds if backend returns them in `/network/topology` response.

## 3. Resizable Panels (FIXED ✅)

**Problem:** CSS `resize` property showed cursor but didn't work in flexbox layout.

**Solution:** Implemented proper JavaScript drag handlers:

### Store (`simulationStore.js`):
- Added `panelSizes` state: `{ leftSidebar: 280, rightPanel: 300, bottomPanel: 220 }`
- Added `setPanelSize(panel, size)` action

### App Component (`App.js`):
- Added resize state: `resizing`, `resizeStartRef`
- Implemented handlers:
  - `handleResizeStart(panel, e)` - captures initial position
  - `handleResizeMove(e)` - calculates new size with constraints
  - `handleResizeEnd()` - cleanup
- Applied dynamic styles to panels: `style={{ width: panelSizes.leftSidebar }}`
- Added resize handle elements with proper event handlers

### Constraints:
- **leftSidebar:** 220-500px
- **rightPanel:** 240-600px
- **bottomPanel:** 120px to 60vh

### Styling (`App.css`):
- Removed non-functional CSS `resize` properties
- Added `.resize-handle-vertical` (4px wide, ew-resize cursor)
- Added `.resize-handle-horizontal` (4px tall, ns-resize cursor)
- Hover effect: blue highlight (#3b82f6)
- Active effect: darker blue (#2563eb)

**Result:** Panels can now be resized by dragging the 4px handles between them.

## Testing Instructions

1. **Start Backend:**
   ```bash
   cd api
   python main.py
   ```

2. **Start Frontend:**
   ```bash
   npm start
   ```

3. **Test Edge Labels:**
   - Initialize simulation (5 banks)
   - Start simulation
   - Observe edges showing current exposure amounts
   - Verify labels update as transactions occur
   - Check Activity Log shows same transactions

4. **Test Resize:**
   - Hover over 4px gap between left sidebar and center panel (cursor should change to ew-resize)
   - Drag left/right to resize
   - Try resizing right panel and bottom panel
   - Verify min/max constraints work

5. **Test CCP Nodes:**
   - Backend must return CCP nodes in `/network/topology` response
   - Verify CCPs appear as purple diamonds
   - Verify label shows "CCP-X" format

## Files Modified

- `src/store/simulationStore.js` - Edge activity logic, panel sizes
- `src/App.js` - Resize handlers, dynamic panel sizing
- `src/App.css` - Resize handle styles
- `src/hooks/useSimulationControl.js` - CCP node ingestion
- `src/systems/cytoscapeStyles.js` - CCP node styles
- `src/components/NetworkGraph.jsx` - Pass node_type field
