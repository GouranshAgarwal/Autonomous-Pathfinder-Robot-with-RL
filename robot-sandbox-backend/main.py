# main.py
import asyncio
import websockets
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from env.robot_env import RobotEnvironment

connected_clients = set()
# Single standalone environment dedicated exclusively to UI Sandbox Inference Mode
inference_env = RobotEnvironment()
model = None

def make_parallel_env(rank, seed=42):
    """
    Factory utility to safely spawn environment instances inside separate CPU processes.
    """
    def _init():
        env = RobotEnvironment()
        # Seed each environment distinctly to maximize variance and exploration
        env.reset(seed=seed + rank)
        return env
    return _init


class LiveTrainingVisualizerCallback(BaseCallback):
    def __init__(self, update_interval=5, main_loop=None):
        super().__init__()
        self.update_interval = update_interval
        self.main_loop = main_loop
        
        # Dedicated sequential visualizer environment
        self.vis_env = RobotEnvironment(render_mode=None)
        self.vis_obs, _ = self.vis_env.reset()

    def _on_step(self) -> bool:
        try:
            # 🚀 CRITICAL FIX: Step the environment on EVERY single engine call.
            # Since global increases by 10, your UI robot now takes 1 parallel step,
            # matching the exact live timeline speed of a single background worker.
            action, _ = self.model.predict(self.vis_obs, deterministic=True)
            self.vis_obs, reward, terminated, truncated, _ = self.vis_env.step(action)
            
            # Handle resetting the visual environment independently when it finishes a path
            if terminated or truncated:
                self.vis_obs, _ = self.vis_env.reset()

            # 🚀 THROTTLE ONLY THE NETWORK: Pass data to React every X steps
            # This protects your browser canvas from an I/O frame rate flood.
            if self.n_calls % self.update_interval == 0:
                payload = {
                    "status": "TRAINING",
                    "robot": self.vis_env.robo_pos.tolist(),
                    "goal": self.vis_env.goal_pos.tolist(),
                    "obstacles": self.vis_env.obstacles.tolist(),
                    "lidar": [float(x) for x in self.vis_env.laser_list],
                    "step": self.vis_env.current_step,
                    "total_timesteps": int(self.model.num_timesteps),
                    "successes": int(self.vis_env.total_successes),
                    "crashes": int(self.vis_env.total_crashes),
                    "reward": round(float(self.vis_env.episode_reward), 2)
                }
                
                if self.main_loop:
                    asyncio.run_coroutine_threadsafe(broadcast_to_all(payload), self.main_loop)
                    
        except Exception as e:
            print(f"Visualization streaming error: {e}")
            
        return True

async def broadcast_to_all(payload):
    if connected_clients:
        msg = json.dumps(payload)
        dead_clients = set()
        for client in connected_clients:
            try:
                if client.open:
                    await client.send(msg)
                else:
                    dead_clients.add(client)
            except Exception:
                dead_clients.add(client)
        
        for client in dead_clients:
            connected_clients.discard(client)

async def handle_client_messages(websocket):
    global model, inference_env
    print(f"Web Browser Node connected: {websocket.remote_address}")
    connected_clients.add(websocket)
    
    try:
        async for message in websocket:
            event = json.loads(message)
            cmd_type = event.get("type")
            
            if cmd_type == "START_TRAINING":
                await broadcast_to_all({"status": "SYSTEM_MSG", "txt": "Spawning 10 Headless Cores + 1 Visualizer Thread..."})
                
                # Capture the current running asyncio network loop
                current_loop = asyncio.get_running_loop()
                
                def run_training():
                    global model
                    num_cpu = 10 
                    print(f"Spawning {num_cpu} headless training environments...")
                    vec_env = SubprocVecEnv([make_parallel_env(i) for i in range(num_cpu)])
                    
                    model = PPO(
                        "MlpPolicy", 
                        vec_env, 
                        verbose=0, 
                        learning_rate=3e-4, 
                        n_steps=512, 
                        batch_size=128,
                        n_epochs=10,
                        gamma=0.99
                    )
                    
                    # Pass the network loop handle so the callback can stream safely
                    # Set update_interval to 5 or 10 to balance backend speed and UI fluidity
                    visualizer_callback = LiveTrainingVisualizerCallback(update_interval=5, main_loop=current_loop)
                    
                    model.learn(total_timesteps=100000, callback=visualizer_callback)
                    model.save("ppo_interactive_navigator")
                    vec_env.close()
                    print("Parallel optimization complete.")

                # Fire up the training thread cleanly without blocking the socket server
                await asyncio.to_thread(run_training)
                await broadcast_to_all({"status": "TRAINING_FINISHED"})
            elif cmd_type == "RUN_INFERENCE":
                if model is None:
                    try: 
                        model = PPO.load("ppo_interactive_navigator")
                    except Exception: 
                        continue
                
                obs = inference_env.set_interactive_layout(
                    robot_pos=event["robot"],
                    goal_pos=event["goal"],
                    obstacles_list=event["obstacles"]
                )
                
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = inference_env.step(action)
                    done = terminated or truncated
                    
                    await broadcast_to_all({
                        "status": "NAVIGATING",
                        "robot": inference_env.robo_pos.tolist(),
                        "goal": inference_env.goal_pos.tolist(),
                        "obstacles": inference_env.obstacles.tolist(),
                        "lidar": [float(x) for x in inference_env.laser_list],
                        "step": inference_env.current_step,
                        "successes": int(inference_env.total_successes),
                        "crashes": int(inference_env.total_crashes),
                        "reward": round(float(inference_env.episode_reward), 2)
                    })
                    await asyncio.sleep(0.04)
                
                await broadcast_to_all({"status": "NAVIGATION_DONE"})

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)

async def main():
    global loop
    loop = asyncio.get_running_loop()
    print("Launching Integrated Control Broker Core on ws://localhost:8000")
    async with websockets.serve(handle_client_messages, "localhost", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())