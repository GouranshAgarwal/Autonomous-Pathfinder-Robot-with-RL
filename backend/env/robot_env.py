# env/robot_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RobotEnvironment(gym.Env):
    def __init__(self):
        super().__init__()

        self.arena_size = 10.0
        self.lidar_max_distance = 5.0
        self.max_vel = 1.0
        self.current_step = 0  
        self.dt = 0.1  # Unified time step for kinematic consistency

        self.cylindrical_obstacles = np.empty((0, 3), dtype=np.float32)
        self.laser_list = [self.lidar_max_distance] * 12

        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.goal = np.array([0.0, 0.0], dtype=np.float32)

        self.robo_radius = 0.2
        self.goal_radius = 0.2

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.theta = 0.0

        self.lidar_offsets = np.linspace(-np.pi, np.pi, 12, endpoint=False, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Extended observation bounds matching your output dimensions (16 elements total)
        low_bound = np.array(
            [-self.arena_size * 2.5] * 2 +
            [-self.max_vel, -1.0] +
            [0.0] * 12,
            dtype=np.float32
        )
        high_bound = np.array(
            [self.arena_size * 2.5] * 2 +
            [self.max_vel, 1.0] +
            [self.lidar_max_distance] * 12,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=low_bound, high=high_bound, shape=(16,), dtype=np.float32
        )

        # Precompute the standard range of steps along each LiDAR ray to save CPU allocation cycles
        self._lidar_steps = np.linspace(self.robo_radius, self.lidar_max_distance, 25, dtype=np.float32)

    def _random_coord_in_arena(self):
        return np.random.uniform(
            -self.arena_size + 0.2, self.arena_size - 0.2, size=(2,)
        ).astype(np.float32)

    def _is_safe(self, pos):
        # 🔥 OPTIMIZED: Vectorized boundary check
        if np.any(np.abs(pos) > self.arena_size):
            return False

        if len(self.cylindrical_obstacles) == 0:
            return True

        # 🔥 OPTIMIZED: Blazing fast matrix distance calculation
        obs_centers = self.cylindrical_obstacles[:, :2]
        obs_radii = self.cylindrical_obstacles[:, 2]
        
        distances = np.linalg.norm(obs_centers - pos, axis=1)
        if np.any(distances < (obs_radii + self.robo_radius)):
            return False
        return True
    
    def _compute_lidar(self):
        # 🔥 HIGHLY OPTIMIZED: Complete extraction of nested Python loops using matrix broadcasting
        beam_angles = self.theta + self.lidar_offsets  # Shape: (12,)
        
        # Build 2D grid vectors via outer products (12, 1) * (1, 25)
        cos_components = np.cos(beam_angles)[:, np.newaxis] * self._lidar_steps[np.newaxis, :]
        sin_components = np.sin(beam_angles)[:, np.newaxis] * self._lidar_steps[np.newaxis, :]
        
        # Broadcast offsets directly against the robot's coordinates -> Shape: (12, 25, 2)
        points_x = self.pos[0] + cos_components
        points_y = self.pos[1] + sin_components
        ray_points = np.stack([points_x, points_y], axis=-1)

        # Matrix Boundary Collision Checks -> Shape: (12, 25)
        hit_boundary = np.any(np.abs(ray_points) > self.arena_size, axis=-1)

        # Matrix Obstacle Collision Checks via Broadcasting
        if len(self.cylindrical_obstacles) > 0:
            obs_centers = self.cylindrical_obstacles[:, :2]  # Shape: (N, 2)
            obs_radii = self.cylindrical_obstacles[:, 2]      # Shape: (N,)

            # Reshape tensors to calculate every single intersection combination simultaneously
            # (12, 25, 1, 2) minus (1, 1, N, 2)
            delta_matrix = ray_points[:, :, np.newaxis, :] - obs_centers[np.newaxis, np.newaxis, :, :]
            distance_matrix = np.linalg.norm(delta_matrix, axis=-1)  # Shape: (12, 25, N)
            
            hit_obstacle = np.any(distance_matrix <= obs_radii[np.newaxis, np.newaxis, :], axis=-1)  # Shape: (12, 25)
        else:
            hit_obstacle = np.zeros((12, 25), dtype=bool)

        # Combine collision maps
        all_collisions = hit_obstacle | hit_boundary  # Shape: (12, 25)

        # Determine the earliest true hit index along the steps axis
        has_hit = np.any(all_collisions, axis=1)
        first_hit_indices = np.argmax(all_collisions, axis=1)

        # Map index hits back to their actual spatial distances
        self.laser_list = np.where(
            has_hit, 
            self._lidar_steps[first_hit_indices], 
            self.lidar_max_distance
        ).astype(np.float32).tolist()

    def _get_obs(self):
        self._compute_lidar()
        goal_vector = self.goal - self.pos

        c, s = np.cos(self.theta), np.sin(self.theta)
        goal_forward = c * goal_vector[0] + s * goal_vector[1]
        goal_sideways = -s * goal_vector[0] + c * goal_vector[1]

        return np.concatenate([
            np.array([goal_forward, goal_sideways], dtype=np.float32),
            np.array([self.linear_velocity, self.angular_velocity], dtype=np.float32),
            np.array(self.laser_list, dtype=np.float32)
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.theta = 0.0

        # 🕹️ EVALUATION OVERRIDE: Check if custom map options are specified by the UI
        if options and "testing_config" in options:
            config = options["testing_config"]
            
            # Map configuration properties natively into numpy structures
            self.pos = np.array(config["start_pos"], dtype=np.float32)
            self.goal = np.array(config["goal_pos"], dtype=np.float32)
            self.theta = float(config.get("start_theta", 0.0))
            
            obstacles_list = config.get("obstacles", [])
            if len(obstacles_list) > 0:
                self.cylindrical_obstacles = np.array(obstacles_list, dtype=np.float32)
            else:
                self.cylindrical_obstacles = np.empty((0, 3), dtype=np.float32)
        else:
            # 🎲 TRAINING DEFAULT: Fall back to random procedural generation
            num_obstacles = np.random.randint(4, 7)
            self.cylindrical_obstacles = np.array([
                [z[0], z[1], np.random.uniform(0.4, 1.2)] 
                for z in [self._random_coord_in_arena() for _ in range(num_obstacles)]
            ], dtype=np.float32)

            while True:
                robo_pos = self._random_coord_in_arena()
                if self._is_safe(robo_pos):
                    self.pos = robo_pos
                    break

            while True:
                goal_pos = self._random_coord_in_arena()
                if self._is_safe(goal_pos) and np.linalg.norm(goal_pos - self.pos) > 4.0:
                    self.goal = goal_pos
                    break

        return self._get_obs(), {}
    
    
    def step(self, action):
        reward = -0.05  
        terminated = False
        truncated = False
        self.current_step += 1

        if self.current_step >= 500:
            truncated = True

        old_goal_dis = np.linalg.norm(self.goal - self.pos)

        # Explicit Euler Integration 
        self.linear_velocity += action[0] * self.dt
        self.linear_velocity = np.clip(self.linear_velocity, 0.0, self.max_vel) 

        self.angular_velocity += action[1] * self.dt
        self.angular_velocity = np.clip(self.angular_velocity, -1.0, 1.0)

        self.theta += self.angular_velocity * self.dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        self.linear_velocity *= 0.98
        self.angular_velocity *= 0.95

        dx = np.cos(self.theta) * self.linear_velocity * self.dt
        dy = np.sin(self.theta) * self.linear_velocity * self.dt
        new_pos = self.pos + np.array([dx, dy], dtype=np.float32)

        if not self._is_safe(new_pos):
            reward -= 100.0  
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        self.pos = new_pos
        new_goal_dis = np.linalg.norm(self.goal - self.pos)
        
        # 1. Distance Progress Reward
        reward += (old_goal_dis - new_goal_dis) * 10.0

        # 2. Heading Alignment Reward
        goal_vector = self.goal - self.pos
        goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
        angle_error = abs(np.arctan2(np.sin(goal_angle - self.theta), np.cos(goal_angle - self.theta)))
        heading_reward = (np.pi - angle_error) / np.pi
        reward += heading_reward * 0.1  

        # 3. LiDAR Obstacle Avoidance Penalty
        min_lidar = min(self.laser_list)
        safety_clearance = 1.4  
        
        if min_lidar < safety_clearance:
            proximity_severity = (safety_clearance - min_lidar) / (safety_clearance - self.robo_radius)
            reward -= (proximity_severity ** 2) * 0.5

        # Terminal Victory Jackpot
        reached_goal = new_goal_dis <= (self.robo_radius + self.goal_radius)
        if reached_goal:
            reward += 300.0
            terminated = True

        info = {
            "is_success": bool(reached_goal),
            "render_state": {
                "robot": {
                    "x": float(self.pos[0]),
                    "y": float(self.pos[1]),
                    "theta": float(self.theta)
                },
                "goal": {
                    "x": float(self.goal[0]),
                    "y": float(self.goal[1])
                },
                "lidar": [float(x) for x in self.laser_list],
                "obstacles": [
                    {"x": float(obs[0]), "y": float(obs[1]), "radius": float(obs[2])}
                    for obs in self.cylindrical_obstacles
                ]
            }
        }

        return self._get_obs(), float(reward), terminated, truncated, info