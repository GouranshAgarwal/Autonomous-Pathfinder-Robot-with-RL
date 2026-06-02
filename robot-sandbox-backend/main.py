# train.py
import multiprocessing
import asyncio
import queue  
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv  # 🔥 Native multi-core orchestration
from stable_baselines3.common.callbacks import BaseCallback

from env.robot_env import RobotEnvironment

# ------------------------------------------------------------------
# 🧠 MULTI-CORE TELEMETRY CALLBACK (EXTRACTS FROM PROCESS 0)
# ------------------------------------------------------------------
class RealTimeTelemetryCallback(BaseCallback):
    def __init__(self, mp_queue, verbose=0):
        super().__init__(verbose)
        self.mp_queue = mp_queue
        self.step_in_episode = 0
        
        self.session_metrics = {
            "successes": 0,
            "crashes": 0,
            "current_reward": 0.0
        }

    def _on_step(self) -> bool:
        self.step_in_episode += 1
        locals_dict = self.locals
        
        # Extract environment metrics specifically from Core 0 to avoid telemetry flooding
        current_reward = float(locals_dict.get("rewards", [0.0])[0])
        self.session_metrics["current_reward"] += current_reward
        
        dones = locals_dict.get("dones", [False])
        infos = locals_dict.get("infos", [{}])
        
        snapshot_reward = self.session_metrics["current_reward"]
        is_terminal_frame = dones[0]

        if is_terminal_frame:
            # Evaluate using process 0's explicit info flags
            if infos[0].get("is_success", False):
                self.session_metrics["successes"] += 1
            else:
                self.session_metrics["crashes"] += 1
            
            self.step_in_episode = 0
            self.session_metrics["current_reward"] = 0.0

        # Frame skip check (samples Core 0 every 5 steps, or forces update on terminal event)
        if self.num_timesteps % 5 == 0 or is_terminal_frame:
            # Extract the raw pre-packaged telemetry from Core 0 across the process barrier
            render_data = infos[0].get("render_state", None)
            
            if render_data:
                try:
                    # Construct structural payload natively without object references
                    payload = {
                        "robot": render_data["robot"],
                        "goal": render_data["goal"],
                        "lidar": render_data["lidar"],
                        "obstacles": render_data["obstacles"],
                        "status": "TRAINING",
                        "total_timesteps": int(self.num_timesteps),
                        "step": int(self.step_in_episode),
                        "successes": int(self.session_metrics["successes"]),
                        "crashes": int(self.session_metrics["crashes"]),
                        "reward": float(snapshot_reward)
                    }
                    
                    if self.mp_queue.full():
                        try:
                            self.mp_queue.get_nowait()
                        except Exception:
                            pass
                    
                    self.mp_queue.put_nowait(payload)
                    
                except Exception as e:
                    print(f"Error transferring telemetry frame: {e}")
                
        return True

# ------------------------------------------------------------------
# ⚙️ PARALLELIZED TRAINING CORE (MAX CPU CONSUMPTION WORKER)
# ------------------------------------------------------------------
def make_env():
    """Helper function to cleanly instantiate environments inside child processes."""
    return RobotEnvironment()

def run_rl_training(telemetry_queue):
    # Detect available CPU threads
    cpu_cores = multiprocessing.cpu_count()
    print(f"🤖 [Child Process] Spawning {cpu_cores} parallel environment instances via SubprocVecEnv...")
    
    # Vectorize across all CPU Cores completely isolated from FastAPI
    env = SubprocVecEnv([make_env for _ in range(cpu_cores)])
    
    # Adjust hyperparameters for multi-core vector processing
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,          # Reduced slightly since steps accumulate simultaneously across cores
        batch_size=128,         # Increased batch sizes work better with multi-core environments
        n_epochs=10
    )
    
    print("🚀 All engines clear. Commencing parallel gradient optimizations...")
    telemetry_cb = RealTimeTelemetryCallback(telemetry_queue)
    
    model.learn(total_timesteps=3_000_000, callback=telemetry_cb)
    
    model.save("ppo_kinematic_robot")
    env.close()
    print("💾 Training finalized. Checkpoint saved.")
    telemetry_queue.put({"status": "TRAINING_FINISHED"})

# ------------------------------------------------------------------
# 🌐 FASTAPI ROUTER & PIPELINE (RUNS IN MAIN PROCESS)
# ------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections = set()
running_processes = [] 
telemetry_queue = multiprocessing.Queue(maxsize=15)

async def telemetry_broadcast_daemon():
    while True:
        if len(active_connections) > 0:
            try:
                payload = telemetry_queue.get_nowait()
                dead_connections = set()
                for client in active_connections:
                    try:
                        await client.send_json(payload)
                    except Exception:
                        dead_connections.add(client)
                for client in dead_connections:
                    active_connections.remove(client)
            except queue.Empty:
                pass  
        await asyncio.sleep(0.002) # Faster responsive frame yield

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telemetry_broadcast_daemon())

@app.on_event("shutdown")
async def shutdown_event():
    print("\n🛑 Server shutting down. Stopping active processes...")
    for p in running_processes:
        if p.is_alive():
            p.terminate()
            p.join()
    print("✨ Process pool clear.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    print(f"🔌 Visualizer Node Active: {len(active_connections)}")
    
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "START_TRAINING":
                print("⚡ UI Signal verified. Spinning multi-core training node...")
                
                training_process = multiprocessing.Process(
                    target=run_rl_training, 
                    args=(telemetry_queue,), 
                    daemon=False
                )
                training_process.start()
                running_processes.append(training_process)
                
            elif data.get("type") == "START_TESTING":
                print("🕹️ Custom map layout received. Initiating evaluation inference...")
                config_payload = data.get("config")
                
                try:
                    # 1. Load your newly generated model checkpoint
                    eval_model = PPO.load("ppo_kinematic_robot")
                    
                    # 2. Spin up a separate evaluation sandbox environment 
                    eval_env = RobotEnvironment()
                    
                    # 3. Inject user properties directly through the custom options hook
                    obs, _ = eval_env.reset(options={"testing_config": config_payload})
                    
                    terminated = False
                    truncated = False
                    
                    while not (terminated or truncated):
                        # Calculate deterministic predictions for smooth navigation paths
                        action, _ = eval_model.predict(obs, deterministic=True)
                        
                        # Execute model physics parameters 
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        
                        # Extract the high-fidelity prepackaged rendering payload
                        render_frame = info.get("render_state", {})
                        
                        # Append structural metadata metrics for UI state tracking
                        render_frame["status"] = "TESTING"
                        render_frame["terminated"] = bool(terminated)
                        render_frame["truncated"] = bool(truncated)
                        render_frame["current_step"] = int(eval_env.current_step)
                        
                        # Transmit frame packet down the primary socket pipe
                        await websocket.send_json(render_frame)
                        
                        # Throttle steps to exactly 100ms to preserve perfect visual tracking matching dt=0.1
                        await asyncio.sleep(0.1)
                        
                    print("🏁 Evaluation routine finalized cleanly.")
                    
                except Exception as eval_err:
                    print(f"Failed to run model evaluation step loop: {eval_err}")
                    await websocket.send_json({"status": "ERROR", "message": str(eval_err)})
                
    except Exception as e:
        print(f"Disconnecting visualizer node... Context: {e}")
    finally:
        active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)