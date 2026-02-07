import React from 'react';
import useSimulationStore from '../store/simulationStore';
import useSimulationControl from '../hooks/useSimulationControl';
import { Play, Pause, SkipForward, RotateCcw } from 'lucide-react';
import './ControlBar.css';

/**
 * Control Bar — Play/Pause/Step/Reset + timestep display.
 * Buttons disabled contextually based on simStatus.
 */
export default function ControlBar() {
  const simStatus = useSimulationStore((s) => s.simStatus);
  const timestep = useSimulationStore((s) => s.timestep);
  const selectedScenario = useSimulationStore((s) => s.selectedScenario);
  const { play, pause, step, reset } = useSimulationControl();

  const isIdle = simStatus === 'idle';
  const isRunning = simStatus === 'running';
  const isPaused = simStatus === 'paused';
  const isDone = simStatus === 'done';
  const canRun = selectedScenario && (isIdle || isDone);
  const canPause = isRunning;
  const canStep = isPaused;
  const canReset = !isIdle;

  return (
    <div className="control-bar">
      <div className="control-left">
        {/* Primary action: Run or Pause */}
        {isRunning ? (
          <button
            className="ctrl-btn ctrl-pause"
            onClick={pause}
            disabled={!canPause}
          >
            <Pause size={14} />
            Pause
          </button>
        ) : (
          <button
            className="ctrl-btn ctrl-play"
            onClick={play}
            disabled={!canRun}
            title={!selectedScenario ? 'Select a scenario first' : ''}
          >
            <Play size={14} />
            {isDone ? 'Restart' : 'Run Simulation'}
          </button>
        )}

        {/* Step (only when paused) */}
        <button
          className="ctrl-btn ctrl-step"
          onClick={step}
          disabled={!canStep}
        >
          <SkipForward size={14} />
          Step
        </button>

        {/* Reset */}
        <button
          className="ctrl-btn ctrl-reset"
          onClick={reset}
          disabled={!canReset}
        >
          <RotateCcw size={14} />
          Reset
        </button>
      </div>

      <div className="control-right">
        {/* Status indicator */}
        <div className={`status-badge status-${simStatus}`}>
          <span className="status-dot" />
          {simStatus.charAt(0).toUpperCase() + simStatus.slice(1)}
        </div>

        {/* Timestep counter */}
        {timestep > 0 && (
          <div className="timestep-display">
            <span className="timestep-label">Step</span>
            <span className="timestep-value">{timestep}</span>
          </div>
        )}
      </div>
    </div>
  );
}
