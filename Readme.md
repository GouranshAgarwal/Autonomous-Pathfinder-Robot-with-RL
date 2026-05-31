# Autonomous Pathfinder Robot with Reinforcement Learning

## 🎯 Project Overview

This project implements an autonomous robot navigation system powered by Reinforcement Learning. The robot learns to navigate complex environments, avoid obstacles, and reach target destinations through continuous training and reward-based feedback mechanisms.

---

## 📋 What We Are Doing

- **Autonomous Robot Navigation**: Building an RL-based agent that learns optimal pathfinding strategies
- **Environment Simulation**: Creating a sandbox environment to train and test the robot's navigation capabilities
- **Real-time Telemetry**: Implementing WebSocket-based telemetry for live monitoring and visualization
- **Model Training**: Developing and refining the RL model to improve navigation behavior and obstacle avoidance

---

## 🔄 Approach

### Architecture
- **Decoupled Web Architecture**: Separation of concerns between backend simulation engine and frontend visualization
  - **Backend** (`robot-sandbox-backend`): RL training engine, environment simulation, and robot control logic
  - **Frontend** (`robot-sandbox-ui`): Real-time visualization and telemetry dashboard

### Reinforcement Learning Strategy
- **State Space**: Robot position, velocity, sensor readings (distance to obstacles and target)
- **Action Space**: Movement commands (forward, backward, rotate left, rotate right)
- **Reward System**: Incentivizes reaching the goal while penalizing collisions and inefficient paths

### Tech Stack
- **Backend**: Python, Docker (54.5% JavaScript, 35.5% Python, 5.8% CSS, 2.8% Dockerfile, 1.4% HTML)
- **Frontend**: JavaScript/React-based UI with WebSocket integration
- **Communication**: WebSocket protocol for real-time telemetry streaming

---

## 🚀 Advantages of Decoupled Architecture

### Modularity & Maintainability
- **Independent Development**: Backend training logic and frontend UI can be developed and deployed separately
- **Scalability**: Easy to scale individual components without affecting others
- **Testing**: Isolated testing for each module improves code quality

### Versatility
- **Pluggable Components**: Different RL algorithms can be tested by only modifying the backend
- **Reusable UI**: The telemetry dashboard can visualize multiple robot simulations simultaneously
- **Flexible Deployment**: Backend can run on high-compute servers; frontend on lightweight clients

### Decoupled WebSocket Telemetry
- **Real-time Monitoring**: Live streaming of robot metrics without tight coupling between systems
- **Asynchronous Communication**: Non-blocking telemetry prevents training lag
- **Multiple Clients**: Multiple front-end instances can connect to a single backend simultaneously
- **Easy Integration**: WebSocket APIs can be integrated with external monitoring tools or custom dashboards

---

## 📊 Current Observations & Known Issues

### Goal-Reaching Behavior
**Issue**: The model approaches the goal quickly but **stops at a certain distance before reaching it**
- **Root Cause**: Imbalanced reward-penalty system
- **Details**: The penalty for collision may be too aggressive or the goal reward too weak, causing the agent to exhibit overly cautious behavior near the target
- **Impact**: Robot fails to complete successful goal captures

### Obstacle Avoidance Behavior
**Issue**: Inconsistent obstacle handling
- **Straight-line Obstacles**: If an obstacle is directly in the path, the robot **collides** instead of maneuvering around it
- **Angled Obstacles**: Robot successfully avoids obstacles that are only slightly in the way
- **Root Cause**: Insufficient penalty diversity and reward shaping for obstacle-specific scenarios

---

## 🔧 Future Improvements

### Short-term
- [ ] Fine-tune reward-penalty ratios for balanced goal-seeking behavior
- [ ] Implement separate reward weights for direct vs. angled obstacles
- [ ] Add reward shaping to encourage earlier obstacle detection and response

### Medium-term
- [ ] Integrate advanced RL algorithms (PPO, DQN improvements)
- [ ] Expand environment complexity (multiple obstacles, dynamic targets)
- [ ] Implement transfer learning for faster convergence

### Long-term
- [ ] Real robot integration and deployment
- [ ] Multi-agent coordination
- [ ] Advanced path planning with dynamic re-routing

---

## 📦 Directory Structure

```
Autonomous-Pathfinder-Robot-with-RL/
├── robot-sandbox-backend/      # RL engine & simulation
├── robot-sandbox-ui/           # Web-based telemetry dashboard
└── Readme.md
```

---

## 🚀 Getting Started

*Detailed setup instructions will be added to respective backend and frontend README files.*

---

## 📝 License

Open source project - contributions welcome!

---

**Built with ❤️ for autonomous robotics and reinforcement learning**
