// app.jsx
import React, { useEffect, useState, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';

export default function InteractiveRobotSandbox() {
  const [serverStatus, setServerStatus] = useState("IDLE"); // IDLE, TRAINING, READY, NAVIGATING
  const [totalTimesteps, setTotalTimesteps] = useState(0);
  const [placementMode, setPlacementMode] = useState("VIEW"); // VIEW, OBSTACLE, GOAL, ROBOT
  
  const [robotPos, setRobotPos] = useState([0, 0]);
  const [goalPos, setGoalPos] = useState([5, 5]);
  const [obstacles, setObstacles] = useState([]);
  const [lidar, setLidar] = useState([]);
  const [simStep, setSimStep] = useState(0);

  const socketRef = useRef(null);
  const ARENA_SIZE = 10.0;

  const [successes, setSuccesses] = useState(0);
  const [crashes, setCrashes] = useState(0);
  const [reward, setReward] = useState(0);

  useEffect(() => {
    socketRef.current = new WebSocket("ws://localhost:8000");
    
    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.status === "TRAINING") {
        setServerStatus("TRAINING");
        setRobotPos(data.robot);
        setGoalPos(data.goal);
        setObstacles(data.obstacles);
        setLidar(data.lidar);
        setTotalTimesteps(data.total_timesteps);
        // FIX: Synchronized step counter state updates during active training iterations
        setSimStep(data.step); 
        setSuccesses(data.successes);
        setCrashes(data.crashes);
        setReward(data.reward);
      } else if (data.status === "TRAINING_FINISHED") {
        setServerStatus("READY");
        setPlacementMode("VIEW");
        alert("Parallel Vector Training Complete! Entered Sandbox Testing Mode.");
      } else if (data.status === "NAVIGATING") {
        setServerStatus("NAVIGATING");
        setRobotPos(data.robot);
        // FIX: Enforce deterministic orientation layout syncing for inference pipelines
        setGoalPos(data.goal);
        setObstacles(data.obstacles);
        setLidar(data.lidar);
        setSimStep(data.step);
        setSuccesses(data.successes);
        setCrashes(data.crashes);
        setReward(data.reward);
      } else if (data.status === "NAVIGATION_DONE") {
        setServerStatus("READY");
        setLidar([]);
      }
    };

    return () => { if(socketRef.current) socketRef.current.close() };
  }, []);

  const handleSceneClick = (e) => {
    if (serverStatus === "TRAINING" || serverStatus === "NAVIGATING") return;
    if (placementMode === "VIEW") return;
    
    e.stopPropagation();

    const clickX = e.point.x;
    const clickY = -e.point.z;

    if (Math.abs(clickX) > ARENA_SIZE || Math.abs(clickY) > ARENA_SIZE) return;

    if (placementMode === "OBSTACLE") {
      const radius = 0.4 + Math.random() * 0.6;
      setObstacles([...obstacles, [clickX, clickY, radius]]);
    } else if (placementMode === "GOAL") {
      setGoalPos([clickX, clickY]);
      setPlacementMode("VIEW");
    } else if (placementMode === "ROBOT") {
      setRobotPos([clickX, clickY]);
      setPlacementMode("VIEW");
    }
  };

  const clearArena = () => {
    setObstacles([]);
    setLidar([]);
    setSimStep(0);
  };

  const startTrainingOnBackend = () => {
    // Clear out stale UI layout counters locally to provide reactive visual cues
    setSuccesses(0);
    setCrashes(0);
    setReward(0);
    setSimStep(0);
    setTotalTimesteps(0);
    socketRef.current.send(JSON.stringify({ type: "START_TRAINING" }));
  };

  const triggerRobotNavigation = () => {
    setPlacementMode("VIEW");
    socketRef.current.send(JSON.stringify({
      type: "RUN_INFERENCE",
      robot: robotPos,
      goal: goalPos,
      obstacles: obstacles
    }));
  };

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#121212', color: '#fff', fontFamily: 'sans-serif' }}>
      
      <div style={{ width: '320px', padding: '20px', backgroundColor: '#1a1a1a', borderRight: '2px solid #2d2d2d', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <h3>Control Panel</h3>
        <div>System Status: <strong style={{ color: serverStatus === "TRAINING" ? "#ffaa00" : "#00ff66" }}>{serverStatus}</strong></div>

        {/* Updated Dashboard: Indicates that tallies reflect the cumulative 10 parallel CPU workers */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', background: '#222', padding: '12px', borderRadius: '6px', border: '1px solid #333' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#888', letterSpacing: '0.05em' }}>🏆 TOTAL SUCCESSES</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#2ecc71' }}>{successes}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#888', letterSpacing: '0.05em' }}>💥 TOTAL CRASHES</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#e74c3c' }}>{crashes}</div>
          </div>
          <div style={{ gridColumn: 'span 2', textAlign: 'center', borderTop: '1px solid #333', paddingTop: '8px', marginTop: '4px' }}>
            <div style={{ fontSize: '0.75rem', color: '#888', letterSpacing: '0.05em' }}>💰 CORE 0 REWARD TALLY</div>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: reward >= 0 ? '#f1c40f' : '#e67e22' }}>{reward}</div>
          </div>
          <div style={{ gridColumn: 'span 2', textAlign: 'center', fontSize: '0.7rem', color: '#555', fontStyle: 'italic' }}>
            Metrics aggregated across 10 core threads
          </div>
        </div>

        {serverStatus === "IDLE" && (
          <button onClick={startTrainingOnBackend} style={{ padding: '12px', background: '#e67e22', border: 'none', color: '#fff', fontWeight: 'bold', cursor: 'pointer', borderRadius: '4px' }}>
            🚀 Start Live Training Run
          </button>
        )}

        {serverStatus === "TRAINING" && (
          <div style={{ background: '#2c3e50', padding: '12px', borderRadius: '4px', borderLeft: '4px solid #3498db' }}>
            🔄 Optimizing Policy Weights... <br />
            <div style={{ marginTop: '8px', fontSize: '0.9rem' }}>
              <strong>Global Engine Step:</strong> {totalTimesteps.toLocaleString()} / 100,000
            </div>
            <div style={{ fontSize: '0.9rem', marginTop: '4px' }}>
              <strong>Current Run Step:</strong> {simStep}
            </div>
          </div>
        )}

        {(serverStatus === "READY" || serverStatus === "NAVIGATING") && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <h4>Sandbox Tools</h4>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: '#222', padding: '12px', borderRadius: '6px', border: '1px solid #333' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: placementMode === "VIEW" ? "#3498db" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "VIEW"} onChange={() => setPlacementMode("VIEW")} disabled={serverStatus === "NAVIGATING"} /> 
                🔍 View Mode (Orbit Scene)
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: placementMode === "OBSTACLE" ? "#e74c3c" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "OBSTACLE"} onChange={() => setPlacementMode("OBSTACLE")} disabled={serverStatus === "NAVIGATING"} /> 
                ➕ Click to Add Obstacle
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: placementMode === "GOAL" ? "#2ecc71" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "GOAL"} onChange={() => setPlacementMode("GOAL")} disabled={serverStatus === "NAVIGATING"} /> 
                🎯 Click to Move Goal
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: placementMode === "ROBOT" ? "#3498db" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "ROBOT"} onChange={() => setPlacementMode("ROBOT")} disabled={serverStatus === "NAVIGATING"} /> 
                🤖 Click to Move Robot
              </label>
            </div>
            
            <button onClick={triggerRobotNavigation} disabled={serverStatus === "NAVIGATING"} style={{ marginTop: '10px', padding: '12px', background: '#2ecc71', border: 'none', color: '#fff', fontWeight: 'bold', cursor: 'pointer', borderRadius: '4px' }}>
              🎯 Test Agent Pathfinding
            </button>
            <button onClick={clearArena} disabled={serverStatus === "NAVIGATING"} style={{ padding: '8px', background: '#c0392b', border: 'none', color: '#fff', cursor: 'pointer', borderRadius: '4px' }}>
              🗑️ Clear Sandbox Layout
            </button>
            <div style={{ fontSize: '0.9rem', color: '#aaa', marginTop: '10px' }}>Inference Trajectory Index: {simStep}</div>
          </div>
        )}
      </div>

      <div style={{ flexGrow: 1, position: 'relative' }}>
        <Canvas camera={{ position: [0, 15, 15], fov: 45 }}>
          <color attach="background" args={["#141414"]} />
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 20, 5]} intensity={1.0} />
          <OrbitControls maxPolarAngle={Math.PI / 2 - 0.05} />

          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} onPointerDown={handleSceneClick}>
            <planeGeometry args={[20, 20]} />
            <meshBasicMaterial transparent opacity={0} />
          </mesh>

          <Grid cellSize={1} cellThickness={1} cellColor="#2c2c2c" sectionSize={5} sectionThickness={1.5} sectionColor="#444" fadeDistance={35} infiniteGrid />

          <BoundaryWalls arenaSize={ARENA_SIZE} height={1.5} />
          
          <mesh position={[goalPos[0], 0.02, -goalPos[1]]}>
            <cylinderGeometry args={[0.2, 0.2, 0.04, 32]} />
            <meshStandardMaterial color="#00ff66" emissive="#00ff66" emissiveIntensity={0.4} />
          </mesh>

          {obstacles.map((obs, index) => (
            <mesh key={index} position={[obs[0], 1.0, -obs[1]]}>
              <cylinderGeometry args={[obs[2], obs[2], 2.0, 24]} />
              <meshStandardMaterial color="#e74c3c" roughness={0.4} />
            </mesh>
          ))}

          {/* FIX: Included totalTimesteps within the dynamic key to guarantee clear buffer compilation sweeps */}
          {lidar.length > 0 && (
            <lineSegments key={`lidar-${totalTimesteps}-${simStep}-${robotPos[0]}-${robotPos[1]}`}>
              <bufferGeometry>
                <float32BufferAttribute
                  attach="attributes-position"
                  args={[
                    new Float32Array(
                      lidar.flatMap((len, idx) => {
                        const angle = (idx / lidar.length) * 2 * Math.PI;
                        const hitX = robotPos[0] + Math.cos(angle) * len;
                        const hitY = robotPos[1] + Math.sin(angle) * len;
                        return [robotPos[0], 0.2, -robotPos[1], hitX, 0.2, -hitY];
                      })
                    ),
                    3
                  ]}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#f1c40f" transparent opacity={0.5} />
            </lineSegments>
          )}

          <mesh position={[robotPos[0], 0.2, -robotPos[1]]}>
            <cylinderGeometry args={[0.2, 0.2, 0.4, 32]} />
            <meshStandardMaterial color="#3498db" metalness={0.3} roughness={0.2} />
          </mesh>
        </Canvas>
      </div>
    </div>
  );
}

function BoundaryWalls({ arenaSize = 10.0, height = 1.0 }) {
  const thickness = 0.2;
  const length = arenaSize * 2;

  return (
    <group>
      {/* North Wall */}
      <mesh position={[0, height / 2, -arenaSize]}>
        <boxGeometry args={[length + thickness, height, thickness]} />
        <meshStandardMaterial color="#ff4444" transparent opacity={0.15} roughness={0.1} metalness={0.5} />
      </mesh>
      
      {/* South Wall */}
      <mesh position={[0, height / 2, arenaSize]}>
        <boxGeometry args={[length + thickness, height, thickness]} />
        <meshStandardMaterial color="#ff4444" transparent opacity={0.15} roughness={0.1} metalness={0.5} />
      </mesh>
      
      {/* East Wall */}
      <mesh position={[arenaSize, height / 2, 0]}>
        <boxGeometry args={[thickness, height, length + thickness]} />
        <meshStandardMaterial color="#ff4444" transparent opacity={0.15} roughness={0.1} metalness={0.5} />
      </mesh>
      
      {/* West Wall */}
      <mesh position={[-arenaSize, height / 2, 0]}>
        <boxGeometry args={[thickness, height, length + thickness]} />
        <meshStandardMaterial color="#ff4444" transparent opacity={0.15} roughness={0.1} metalness={0.5} />
      </mesh>

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.005, 0]}>
        <planeGeometry args={[length, length]} />
        <meshStandardMaterial color="#1a1a1a" roughness={0.8} />
      </mesh>
    </group>
  );
}