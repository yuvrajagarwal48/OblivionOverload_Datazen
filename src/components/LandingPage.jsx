import React, { useRef, useMemo, useCallback, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Text, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import useSimulationStore from '../store/simulationStore';
import { Zap, ArrowRight, Network, TrendingUp, Shield, MousePointer2 } from 'lucide-react';
import './LandingPage.css';

/* ─── 3D Network Node (interactive) ─── */
function NetworkNode({ position, color, size, label, onHover }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y =
        position[1] + Math.sin(state.clock.elapsedTime * 0.5 + position[0]) * 0.3;
      // Pulse glow on hover
      const s = hovered ? size * 1.3 : size;
      meshRef.current.scale.lerp(new THREE.Vector3(s, s, s), 0.1);
    }
  });

  return (
    <group>
      <mesh
        ref={meshRef}
        position={position}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          onHover?.(label);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          setHovered(false);
          onHover?.(null);
          document.body.style.cursor = 'default';
        }}
      >
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered ? 0.9 : 0.4}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>
      {/* Node label always visible */}
      <Text
        position={[position[0], position[1] - size * 1.4 - 0.35, position[2]]}
        fontSize={0.28}
        color={hovered ? '#ffffff' : '#94a3b8'}
        anchorX="center"
        anchorY="top"
        font={undefined}
      >
        {label}
      </Text>
    </group>
  );
}

/* ─── 3D Network Edge ─── */
function NetworkEdge({ start, end }) {
  const lineRef = useRef();

  const points = useMemo(() => {
    return [new THREE.Vector3(...start), new THREE.Vector3(...end)];
  }, [start, end]);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints(points);
    return g;
  }, [points]);

  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial color="#4f7df5" opacity={0.25} transparent />
    </line>
  );
}

/* ─── Floating Particles ─── */
function Particles({ count = 200 }) {
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const particles = useMemo(() => {
    return Array.from({ length: count }, () => ({
      position: [
        (Math.random() - 0.5) * 30,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 15,
      ],
      speed: 0.2 + Math.random() * 0.8,
      offset: Math.random() * Math.PI * 2,
    }));
  }, [count]);

  useFrame((state) => {
    if (!meshRef.current) return;
    particles.forEach((p, i) => {
      const t = state.clock.elapsedTime;
      dummy.position.set(
        p.position[0] + Math.sin(t * p.speed + p.offset) * 0.5,
        p.position[1] + Math.cos(t * p.speed * 0.7 + p.offset) * 0.3,
        p.position[2]
      );
      dummy.scale.setScalar(0.02 + Math.sin(t * 2 + p.offset) * 0.01);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial color="#6b9fff" transparent opacity={0.5} />
    </instancedMesh>
  );
}

/* ─── 3D Scene ─── */
function Scene({ onNodeHover }) {
  const nodes = useMemo(() => [
    { pos: [-4, 2, 0], color: '#10b981', size: 0.5, label: 'JPM' },
    { pos: [-2, -1, -1], color: '#3b82f6', size: 0.4, label: 'GS' },
    { pos: [0, 1.5, 1], color: '#f59e0b', size: 0.6, label: 'BOA' },
    { pos: [2, -0.5, 0], color: '#ef4444', size: 0.35, label: 'CS' },
    { pos: [4, 1, -1], color: '#10b981', size: 0.45, label: 'MS' },
    { pos: [-3, 0.5, 1], color: '#8b5cf6', size: 0.3, label: 'WF' },
    { pos: [1, -2, -0.5], color: '#3b82f6', size: 0.5, label: 'DB' },
    { pos: [3, 2.5, 0.5], color: '#f59e0b', size: 0.35, label: 'BCS' },
    { pos: [-1, -1.5, 1], color: '#10b981', size: 0.4, label: 'CITI' },
    { pos: [5, -1.5, -0.5], color: '#ef4444', size: 0.45, label: 'UBS' },
    { pos: [-5, -0.5, 0.5], color: '#3b82f6', size: 0.35, label: 'HSBC' },
    { pos: [0, 3, -1], color: '#8b5cf6', size: 0.3, label: 'BNP' },
  ], []);

  const edges = useMemo(() => [
    [nodes[0].pos, nodes[1].pos], [nodes[1].pos, nodes[2].pos],
    [nodes[2].pos, nodes[3].pos], [nodes[3].pos, nodes[4].pos],
    [nodes[5].pos, nodes[0].pos], [nodes[6].pos, nodes[2].pos],
    [nodes[7].pos, nodes[4].pos], [nodes[8].pos, nodes[1].pos],
    [nodes[9].pos, nodes[3].pos], [nodes[10].pos, nodes[5].pos],
    [nodes[11].pos, nodes[7].pos], [nodes[6].pos, nodes[8].pos],
    [nodes[0].pos, nodes[2].pos], [nodes[4].pos, nodes[9].pos],
  ], [nodes]);

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#7db4ff" />
      <pointLight position={[-10, -5, 5]} intensity={0.6} color="#4f7df5" />

      {nodes.map((node, i) => (
        <Float key={i} speed={1.5} rotationIntensity={0} floatIntensity={0.5}>
          <NetworkNode
            position={node.pos}
            color={node.color}
            size={node.size}
            label={node.label}
            onHover={onNodeHover}
          />
        </Float>
      ))}

      {edges.map((edge, i) => (
        <NetworkEdge key={i} start={edge[0]} end={edge[1]} />
      ))}

      <Particles count={150} />

      <OrbitControls
        enableZoom={true}
        enablePan={false}
        enableRotate={true}
        autoRotate
        autoRotateSpeed={0.6}
        minDistance={6}
        maxDistance={20}
        maxPolarAngle={Math.PI / 1.5}
        minPolarAngle={Math.PI / 4}
      />
    </>
  );
}

/* ─── Landing Page — Split Layout ─── */
function LandingPage() {
  const enterApp = useSimulationStore((s) => s.enterApp);
  const [hoveredNode, setHoveredNode] = useState(null);

  const handleEnter = useCallback(() => {
    enterApp();
  }, [enterApp]);

  return (
    <div className="landing">
      {/* Skip to Login button */}
      <button 
        className="landing-skip-btn" 
        onClick={handleEnter}
        title="Skip to Login"
      >
        Skip to Login →
      </button>

      {/* ── Left: Text Content ── */}
      <div className="landing-left">
        <div className="landing-content">
          <div className="landing-badge">
            <Zap size={14} />
            <span>Powered by MAPPO</span>
          </div>

          <h1 className="landing-title">
            <span className="title-line">Financial Network</span>
            <span className="title-line title-accent">Simulation Engine</span>
          </h1>

          <p className="landing-description">
            Explore systemic risk propagation through multi-agent reinforcement learning.
            Watch banks make real-time decisions — lend, hoard, or fire-sell — as
            contagion spreads through the interbank network.
          </p>

          <div className="landing-features">
            <div className="feature-item">
              <Network size={20} />
              <span>Dynamic Network Graphs</span>
            </div>
            <div className="feature-item">
              <TrendingUp size={20} />
              <span>Real-Time Analytics</span>
            </div>
            <div className="feature-item">
              <Shield size={20} />
              <span>Risk Assessment</span>
            </div>
          </div>

          <button className="landing-cta" onClick={handleEnter}>
            <span>Launch Simulator</span>
            <ArrowRight size={20} />
          </button>

          <div className="landing-footer-inline">
            <span>FinSim-MAPPO v1.0</span>
            <span className="footer-dot">·</span>
            <span>Datathon 2026</span>
          </div>
        </div>
      </div>

      {/* ── Right: Interactive 3D Graph ── */}
      <div className="landing-right">
        <Canvas
          camera={{ position: [0, 0, 12], fov: 60 }}
          dpr={[1, 2]}
          gl={{ antialias: true, alpha: true }}
        >
          <Scene onNodeHover={setHoveredNode} />
        </Canvas>

        {/* Tooltip for hovered node */}
        {hoveredNode && (
          <div className="landing-tooltip">
            <span className="tooltip-dot" />
            <span>{hoveredNode}</span>
          </div>
        )}

        {/* Interaction hint */}
        <div className="landing-hint">
          <MousePointer2 size={14} />
          <span>Drag to rotate · Scroll to zoom · Hover nodes</span>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;
