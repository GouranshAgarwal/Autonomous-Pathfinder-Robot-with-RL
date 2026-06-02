// app.jsx
import React, { useEffect, useState, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';

export default function InteractiveRobotSandbox() {
  const [serverStatus, setServerStatus] = useState("IDLE"); // IDLE, TRAINING, TESTING, CRASHED, etc.
  const [totalTimesteps, setTotalTimesteps] = useState(0);
  const [placementMode, setPlacementMode] = useState("VIEW"); // VIEW, OBSTACLE, GOAL, ROBOT
  
  const [robotState, setRobotState] = useState({
    x: 0,
    y: 0,
    theta: 0,
    linear_velocity: 0,
    angular_velocity: 0
  });

  const [observation, setObservation] = useState({
    goal_forward: 0,
    goal_sideways: 0,
    linear_velocity: 0,
    angular_velocity: 0
  });

  const [goalPos, setGoalPos] = useState({ x: 4.0, y: 3.5 });
  const [obstacles, setObstacles] = useState([]);
  const [lidar, setLidar] = useState([]);
  const [simStep, setSimStep] = useState(0);

  const socketRef = useRef(null);
  const ARENA_SIZE = 10.0;

  const [successes, setSuccesses] = useState(0);
  const [crashes, setCrashes] = useState(0);
  const [reward, setReward] = useState(0);

  useEffect(() => {
    socketRef.current = new WebSocket("ws://localhost:8000/ws");
    
    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.robot) {
        setRobotState({
          x: data.robot.x,
          y: data.robot.y,
          theta: data.robot.theta,
          linear_velocity: data.robot.linear_velocity || 0,
          angular_velocity: data.robot.angular_velocity || 0
        });
      }

      if (data.goal) {
        setGoalPos(data.goal); 
      }

      if (data.obstacles) {
        setObstacles(data.obstacles); 
      }

      if (data.lidar) {
        setLidar(data.lidar);
      }

      setTotalTimesteps(data.total_timesteps || 0);
      setSimStep(data.current_step || data.step_count || data.step || 0); 
      setSuccesses(data.total_successes || data.successes || 0);
      setCrashes(data.total_crashes || data.crashes || 0);
      setReward(data.reward || 0);

      if (data.observation) {
        setObservation(data.observation);
      } else if (data.robot) {
        setObservation({
          goal_forward: data.goal_forward || 0,
          goal_sideways: data.goal_sideways || 0,
          linear_velocity: data.robot.linear_velocity || 0,
          angular_velocity: data.robot.angular_velocity || 0
        });
      }
      
      if (data.status) {
        setServerStatus(data.status);
      }
    };

    return () => { if(socketRef.current) socketRef.current.close() };
  }, []);

  const handleSceneClick = (e) => {
    if (serverStatus !== "IDLE") return; // Block modifications unless explicitly IDLE
    if (placementMode === "VIEW") return;
    
    e.stopPropagation();
    const clickX = e.point.x;
    const clickY = -e.point.z; 

    if (Math.abs(clickX) > ARENA_SIZE || Math.abs(clickY) > ARENA_SIZE) return;

    if (placementMode === "OBSTACLE") {
      const radius = 0.4 + Math.random() * 0.6;
      setObstacles([...obstacles, { x: clickX, y: clickY, radius }]);
    } else if (placementMode === "GOAL") {
      setGoalPos({ x: clickX, y: clickY });
      setPlacementMode("VIEW");
    } else if (placementMode === "ROBOT") {
      setRobotState(prev => ({
        ...prev,
        x: clickX,
        y: clickY,
        theta: 0.0 
      }));
      setPlacementMode("VIEW");
    }
  };

  const clearArena = () => {
    setObstacles([]);
    setLidar([]);
    setSimStep(0);
  };

  const startTrainingOnBackend = () => {
    setSuccesses(0);
    setCrashes(0);
    setReward(0);
    setSimStep(0);
    setTotalTimesteps(0);
    setServerStatus("TRAINING");
    socketRef.current.send(JSON.stringify({ type: "START_TRAINING" }));
  };

  const triggerRobotNavigation = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      setPlacementMode("VIEW");
      setServerStatus("TESTING");
      
      socketRef.current.send(JSON.stringify({
        type: "START_TESTING",
        config: {
          start_pos: [robotState.x, robotState.y],
          start_theta: robotState.theta,
          goal_pos: [goalPos.x, goalPos.y],
          obstacles: obstacles.map(o => [o.x, o.y, o.radius])
        }
      }));
    }
  };

  // UI FIX: Manual breakout function to drop back into edit mode without refreshing
  const resetToIdleMode = () => {
    setServerStatus("IDLE");
    setPlacementMode("VIEW");
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "STOP_TESTING" }));
    }
  };

  const isTestingActive = serverStatus === "TESTING";
  const isEditable = serverStatus === "IDLE";

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#121212', color: '#fff', fontFamily: 'sans-serif', boxSizing: 'border-box' }}>
      
      <div style={{ width: '320px', flexShrink: 0, padding: '20px', backgroundColor: '#1a1a1a', borderRight: '2px solid #2d2d2d', display: 'flex', flexDirection: 'column', gap: '20px', boxSizing: 'border-box', overflowY: 'auto' }}>
        <h3>Control Panel</h3>
        <div style={{ background:"#222", padding:"12px", borderRadius:"6px", border:"1px solid #333" }}>
          <h4>Robot Observation</h4>
          <div>Goal Forward: {observation.goal_forward.toFixed(2)}</div>
          <div>Goal Sideways: {observation.goal_sideways.toFixed(2)}</div>
          <div>Linear Velocity: {robotState.linear_velocity.toFixed(2)} m/s</div>
          <div>Angular Velocity: {robotState.angular_velocity.toFixed(2)} rad/s</div>
        </div>
        
        <div>
          System Status: {' '}
          <strong style={{ 
            color: serverStatus === "TRAINING" ? "#ffaa00" : 
                   serverStatus === "TESTING" ? "#3498db" : 
                   serverStatus === "IDLE" ? "#00ff66" : "#e74c3c" 
          }}>
            {serverStatus}
          </strong>
        </div>

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
            <div style={{ fontSize: '0.75rem', color: '#888', letterSpacing: '0.05em' }}>💰 LIVE CORE REWARD TALLY</div>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: reward >= 0 ? '#f1c40f' : '#e67e22' }}>{reward.toFixed(2)}</div>
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
              <strong>Global Engine Step:</strong> {totalTimesteps.toLocaleString()}
            </div>
            <div style={{ fontSize: '0.9rem', marginTop: '4px' }}>
              <strong>Current Episode Step:</strong> {simStep}
            </div>
          </div>
        )}

        {/* UI FIX 1: Allow sandbox tools panel to stay visible unless actively training */}
        {serverStatus !== "TRAINING" && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <h4>Sandbox Tools</h4>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: '#222', padding: '12px', borderRadius: '6px', border: '1px solid #333' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: !isEditable ? '#555' : placementMode === "VIEW" ? "#3498db" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "VIEW"} onChange={() => setPlacementMode("VIEW")} disabled={!isEditable} /> 
                🔍 View Mode (Orbit Scene)
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: !isEditable ? '#555' : placementMode === "OBSTACLE" ? "#e74c3c" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "OBSTACLE"} onChange={() => setPlacementMode("OBSTACLE")} disabled={!isEditable} /> 
                ➕ Click to Add Obstacle
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: !isEditable ? '#555' : placementMode === "GOAL" ? "#2ecc71" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "GOAL"} onChange={() => setPlacementMode("GOAL")} disabled={!isEditable} /> 
                🎯 Click to Move Goal
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: !isEditable ? '#555' : placementMode === "ROBOT" ? "#3498db" : "#fff" }}>
                <input type="radio" name="tool" checked={placementMode === "ROBOT"} onChange={() => setPlacementMode("ROBOT")} disabled={!isEditable} /> 
                🤖 Click to Move Robot
              </label>
            </div>
            
            {/* UI FIX 2: Contextual button swap to let users force reset/edit positions without window reload */}
            {isEditable ? (
              <button onClick={triggerRobotNavigation} style={{ marginTop: '10px', padding: '12px', background: '#2ecc71', border: 'none', color: '#fff', fontWeight: 'bold', cursor: 'pointer', borderRadius: '4px' }}>
                🎯 Test Agent Pathfinding
              </button>
            ) : (
              <button onClick={resetToIdleMode} style={{ marginTop: '10px', padding: '12px', background: '#c0392b', border: 'none', color: '#fff', fontWeight: 'bold', cursor: 'pointer', borderRadius: '4px' }}>
                🛑 Stop & Reset to Edit Mode
              </button>
            )}

            <button onClick={clearArena} disabled={!isEditable} style={{ padding: '8px', background: isEditable ? '#c0392b' : '#333', color: isEditable ? '#fff' : '#666', border: 'none', cursor: isEditable ? 'pointer' : 'not-allowed', borderRadius: '4px' }}>
              🗑️ Clear Sandbox Layout
            </button>
            <div style={{ fontSize: '0.9rem', color: '#aaa', marginTop: '10px' }}>Inference Trajectory Index: {simStep}</div>
          </div>
        )}
      </div>

      <div style={{ flexGrow: 1, height: '100%', minWidth: 0, position: 'relative', overflow: 'hidden' }}>
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
          
          <mesh position={[goalPos.x, 0.02, -goalPos.y]}>
            <cylinderGeometry args={[0.2, 0.2, 0.04, 32]} />
            <meshStandardMaterial color="#00ff66" emissive="#00ff66" emissiveIntensity={0.4} />
          </mesh>

          {obstacles.map((obs, index) => (
            <mesh key={index} position={[obs.x, 1.0, -obs.y]}>
              <cylinderGeometry args={[obs.radius, obs.radius, 2.0, 24]} />
              <meshStandardMaterial color="#e74c3c" roughness={0.4} />
            </mesh>
          ))}

          <lineSegments key={`lidar-${totalTimesteps}-${simStep}-${robotState.x}-${robotState.y}-${obstacles.length}`}>
            <bufferGeometry>
              {lidar.length > 0 ? (
                <float32BufferAttribute
                  attach="attributes-position"
                  args={[
                    new Float32Array(
                      lidar.flatMap((len, idx) => {
                        const offset = -Math.PI + idx * (2 * Math.PI / lidar.length);
                        const beamAngle = robotState.theta + offset;
                        const hitX = robotState.x + Math.cos(beamAngle) * len;
                        const hitY = robotState.y + Math.sin(beamAngle) * len;
                        return [robotState.x, 0.2, -robotState.y, hitX, 0.2, -hitY];
                      })
                    ),
                    3
                  ]}
                />
              ) : (
                <float32BufferAttribute attach="attributes-position" args={[new Float32Array(0), 3]} />
              )}
            </bufferGeometry>
            <lineBasicMaterial color="#f1c40f" transparent opacity={0.5} />
          </lineSegments>

          <group
            position={[robotState.x, 0.2, -robotState.y]}
            rotation={[0, -robotState.theta, 0]}
          >
            <mesh>
              <cylinderGeometry args={[0.2, 0.2, 0.4, 32]} />
              <meshStandardMaterial color="#3498db" />
            </mesh>

            <mesh position={[0.25, 0, 0]}>
              <boxGeometry args={[0.15, 0.1, 0.1]} />
              <meshStandardMaterial color="#f1c40f" />
            </mesh>
          </group>
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
      <mesh position={[0, height / 2, -arenaSize]}>
        <boxGeometry args={[length + thickness, height, thickness]} />
        <meshStandardMaterial color="#ff4444" transparent opacity={0.15} roughness={0.1} metalness={0.5} />
      </mesh>
      <mesh position={[0, height / 2, arenaSize]}>
        <boxGeometry args={[length + thickness, height, thickness]} />
        <meshStandardMaterial color="#ff4444" transparent opacity={0.15} roughness={0.1} metalness={0.5} />
      </mesh>
      <mesh position={[arenaSize, height / 2, 0]}>
        <boxGeometry args={[thickness, height, length + thickness]} />
        <meshStandardMaterial color="#ff4444" transparent opacity={0.15} roughness={0.1} metalness={0.5} />
      </mesh>
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