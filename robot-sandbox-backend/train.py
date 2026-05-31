# train.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env.robot_env import RobotEnvironment  # FIX: Aligned with the environment directory layout

def make_env(rank, seed=42):
    """
    Utility factory for multiprocess environment initialization.
    """
    def _init():
        # Keep render_mode=None so environments run at pure math speed without network overhead
        env = RobotEnvironment(render_mode=None)
        # Enforce distinct initialization seeding per CPU worker process thread
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    # Maximize training performance by spreading workloads across parallel workers
    # Set to 4 for mid-tier local testing, or scale up to 10 to match your main broker core speed
    num_cpu = 10  
    print(f"Spawning {num_cpu} parallel environment engines via SubprocVecEnv...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Setup PPO agent with optimized hyperparameters matching the sandbox ecosystem
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_warehouse_logs/"
    )

    print("Beginning optimization run across parallel cores...")
    model.learn(total_timesteps=200000)
    
    # FIX: Export trained weights to match the naming convention expected by main.py / app.jsx
    model.save("ppo_interactive_navigator")
    print("Model saved successfully as ppo_interactive_navigator.zip")
    
    # Clean shutdown of background worker sub-processes
    vec_env.close()