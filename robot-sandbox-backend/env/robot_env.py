# env/robot_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RobotEnvironment(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.arena_size = 10.0
        self.robo_radius = 0.2
        self.goal_radius = 0.2
        self.laser_max_distance = 5.0
        
        self.robo_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.obstacles = np.empty((0, 3), dtype=np.float32)
        self.laser_list = [self.laser_max_distance] * 16
        
        # 🚀 FIX: Calibrate observation bounds to accommodate relative vectors safely
        # Max distance across a 10x10 quadrant arena can span up to 20 units delta
        self.low_bounds = np.array(
            [-self.arena_size * 1.5, -self.arena_size * 1.5] +  # Robot absolute X, Y
            [-self.arena_size * 2.5, -self.arena_size * 2.5] +  # Relative Goal Vector delta X, Y
            [0.0] * 16,                                         # 16-channel LIDAR
            dtype=np.float32
        )
        self.high_bounds = np.array(
            [self.arena_size * 1.5, self.arena_size * 1.5] +
            [self.arena_size * 2.5, self.arena_size * 2.5] +
            [self.laser_max_distance * 1.5] * 16,
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_bounds, high=self.high_bounds, shape=(20,), dtype=np.float32)

        self.render_mode = render_mode
        self.current_step = 0
        
        self.total_successes = 0
        self.total_crashes = 0
        self.episode_reward = 0.0

    def _random_in_arena(self):
        return np.random.uniform(-self.arena_size, self.arena_size, size=(2,)).astype(np.float32)
    
    def _is_safe(self, pos):
        if abs(pos[0]) > self.arena_size or abs(pos[1]) > self.arena_size:
            return False
        for obs in self.obstacles:
            center = obs[:2]
            radius = obs[2]
            distance = np.linalg.norm(pos - center)
            if distance < (radius + self.robo_radius):
                return False
        return True
        
    def _compute_lidar(self):
        laser_list = []
        angle_list = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        step_list = np.linspace(self.robo_radius, self.laser_max_distance, 50)
        
        for angle in angle_list:
            for step in step_list:
                offset = np.array([np.cos(angle) * step, np.sin(angle) * step], dtype=np.float32)
                check_point = self.robo_pos + offset
                if not self._is_safe(check_point):
                    laser_list.append(step)
                    break
            else: 
                laser_list.append(self.laser_max_distance)
        self.laser_list = laser_list

    def _get_obs(self):
        self._compute_lidar()
        # 🚀 FIX: Feed the network a relative displacement vector instead of absolute coordinates
        # This provides an explicit directional heading that remains highly granular near the goal
        relative_goal_vector = self.goal_pos - self.robo_pos
        return np.concatenate([self.robo_pos, relative_goal_vector, self.laser_list], dtype=np.float32)

    def set_interactive_layout(self, robot_pos, goal_pos, obstacles_list):
        self.robo_pos = np.array(robot_pos, dtype=np.float32)
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.obstacles = np.array(obstacles_list, dtype=np.float32)
        self.current_step = 0
        self.episode_reward = 0.0
        return self._get_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        
        number_of_obstacles = np.random.randint(5, 12)
        obstacle_list = []
        for _ in range(number_of_obstacles):
            x, y = self._random_in_arena()
            radius = np.random.uniform(0.4, 1.2)
            obstacle_list.append([x, y, radius])

        self.obstacles = np.array(obstacle_list, dtype=np.float32)
        
        while True:
            robo_pos = self._random_in_arena()
            if self._is_safe(robo_pos):
                self.robo_pos = robo_pos
                break
                
        while True:
            goal_pos = self._random_in_arena()
            if self._is_safe(goal_pos) and np.linalg.norm(self.robo_pos - goal_pos) > 4.0:
                self.goal_pos = goal_pos
                break

        return self._get_obs(), {}
        
    def step(self, action):
        truncated = False
        terminated = False
        
        # 🚀 FIX: Increase the step penalty slightly to discourage lingering behaviors
        reward = -0.15 
        self.current_step += 1

        old_dis = np.linalg.norm(self.goal_pos - self.robo_pos)
        new_pos = self.robo_pos + (action*0.25)
        new_dis = np.linalg.norm(new_pos - self.goal_pos)

        # 🚀 FIX: Atomic safety evaluation. Verify the move before updating state coordinates
        if not self._is_safe(new_pos):
            reward -= 120.0
            self.total_crashes += 1
            terminated = True
            self.episode_reward += reward
            return self._get_obs(), reward, terminated, truncated, {}

        # Move committed only if safe
        self.robo_pos = new_pos

        # 🚀 FIX: Balanced Potential Field Reward formulation
        # Dense directional progress combined with an absolute proximity benefit
        progress = old_dis - new_dis
        reward += progress * 20.0     # Incremental movement reward
        reward -= new_dis * 0.2       # Continuous distance penalty (pulls the robot forward)

        # Goal reached check
        if new_dis < (self.robo_radius + self.goal_radius):
            reward += 200.0           # Increased finishing bonus
            self.total_successes += 1
            terminated = True

        if self.current_step >= 200:
            truncated = True
            
        self.episode_reward += reward
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass