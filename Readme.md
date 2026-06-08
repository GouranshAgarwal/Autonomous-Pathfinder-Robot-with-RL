# Autonomous Pathfinder Robot with Reinforcement Learning

An industry-grade, microservice-decoupled simulation platform for autonomous robotic navigation. The agent trains via continuous Proximal Policy Optimization (PPO) inside a high-performance, custom Python environment and streams live operational telemetry to an interactive, hardware-accelerated 3D web dashboard.

<div align="center">
  <a href="https://www.youtube.com/watch?v=Rjy0Xhp2cj4" target="_blank">
    <img src="https://img.youtube.com/vi/Rjy0Xhp2cj4/maxresdefault.jpg" alt="Autonomous Robot RL Simulation Demo" width="100%">
  </a>
</div>

---

## 🛠️ System Architecture & Design Choices

The platform is explicitly split into decoupled microservices to separate heavy algorithmic mathematics from graphic computation layers, minimizing communication blocks and system overhead.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCKER COMPOSE NETWORK                             │
│                                                                             │
│  ┌─────────────────────────┐               ┌─────────────────────────────┐  │
│  │     PYTHON BACKEND      │               │       REACT FRONTEND        │  │
│  │ ┌─────────────────────┐ │               │ ┌─────────────────────────┐ │  │
│  │ │ Custom Gymnasium Env│ │  JSON Stream  │ │   Asynchronous Custom   │ │  │
│  │ │ NumPy Vector Loops  │ │ ────────────> │ │    WebSocket Hooks      │ │  │
│  │ └──────────┬──────────┘ │  (Low-Latency)│ └────────────┬────────────┘ │  │
│  │            │ State Vectors              │              │ State Matrix │  │
│  │            ▼            │               │              ▼              │  │
│  │ ┌─────────────────────┐ │               │ ┌─────────────────────────┐ │  │
│  │ │ PPO Policy Network  │ │               │ │   React-Three-Fiber     │ │  │
│  │ │  (PyTorch Inference)│ │               │ │     (3D Canvas)         │ │  │
│  │ └─────────────────────┘ │               │ └─────────────────────────┘ │  │
│  └─────────────────────────┘               └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **The Simulation Engine (Backend):** Built on a custom **Gymnasium** environment. To guarantee high frame-per-second exploration throughput during training, all spatial calculations—including boundary limits, obstacle proximities, and 12-ray continuous LiDAR tracking—are evaluated natively using **NumPy matrix broadcasting** instead of iterative loops.
* **The Streaming Pipeline (Network):** Telemetry packets (coordinate states, linear/angular velocities, active step rewards, and raw laser arrays) are serialized into light JSON payloads and pushed over an asynchronous duplex **FastAPI WebSocket** channel.
* **The Telemetry Visualization (Frontend):** A Single Page Application built on **React** that feeds incoming WebSocket state frames into custom state hooks, updating an interactive, hardware-accelerated **React-Three-Fiber (Three.js)** canvas at low latency.

---

## 🧠 Algorithmic Bottlenecks & Self-Authored Solutions

### 1. Mathematical Resolution of the Boundary Local Minimum Trap
During early training iterations, the navigation agent suffered from structural paralysis when encountering continuous obstacle configurations. Because the destination coordinates lay directly beyond a solid wall, a traditional inverse-distance reward landscape created an artificial local minimum. The agent maximized rewards by standing perfectly still and staring directly at the target, as turning around or backing away incurred steep immediate negative distance rewards.

To break this mathematical deadlock, the environment reward function was refactored to introduce a **Dynamic Heading Suppression Modifier** linked to an **Exponential Proximity Scaling Penalty**. 

The dense reward calculation per timestep $t$ is formulated as follows:

$$R_t = -0.1 + 10 \cdot (d_{\text{old}} - d_{\text{new}}) + \tilde{R}_{\text{heading}} - P_{\text{proximity}}$$

Where:
* $-0.1$ is a rigid step penalty enforcing time-optimal trajectories and punishing stagnation.
* $d$ represents the Euclidean distance from the robot center to the target coordinate.

The modified heading alignment tracking reward $\tilde{R}_{\text{heading}}$ dynamically chokes out positive reinforcement if the robot faces an obstacle block directly, forcing the policy to seek alternative angles:

$$\tilde{R}_{\text{heading}} = \begin{cases} 
0.2 \cdot \left(\frac{\pi - \theta_{\text{error}}}{\pi}\right) \cdot d_{\min} & \text{if } d_{\min} < 1.0 \\
0.2 \cdot \left(\frac{\pi - \theta_{\text{error}}}{\pi}\right) & \text{otherwise}
\end{cases}$$

The exponential proximity barrier penalty $P_{\text{proximity}}$ outcompetes forward distance motivation as the robot approaches a danger radius, transforming the obstacle boundaries into aggressive repulsors:

$$P_{\text{proximity}} = \begin{cases} 
2.0 \cdot \left(\frac{1.4 - d_{\min}}{1.4 - r_{\text{robot}}}\right)^2 & \text{if } d_{\min} < 1.4 \\
0 & \text{otherwise}
\end{cases}$$

Where $d_{\min}$ represents the minimum distance reading returned across the 12-ray LiDAR array and $r_{\text{robot}}$ is the structural radius of the agent ($0.2\text{m}$).

### 2. Multi-Stage Container Size Optimization
Deep learning pipelines typically inherit massive CUDA, cuDNN, and system development tools, causing standard deployment images to balloon well beyond 4GB. To make this stack cloud-ready for standard, cost-effective instances, two container techniques were deployed:
* **Backend:** Intercepted downstream dependency layers by explicitly pointing the package manager to pre-compiled x86 pure CPU wheel binaries hosted directly by PyTorch, bypassing heavy graphic hardware overhead.
* **Frontend:** Implemented a multi-stage Docker build process. A Node.js container compiles the static production distribution bundles before transferring the compiled directories onto an ultra-lightweight, hardened Nginx Alpine image, purging the heavy node modules directory from the final image context.

---

## ⚙️ Deployment & Execution

The entire platform is fully containerized and orchestrated via Docker Compose.

### Prerequisites
Ensure your host machine has Docker and Docker Compose installed.

### Spin up the Microservices
Clone the repository and execute the following deployment command from the root directory:

```bash
docker-compose up --build
```

Once the compilation layer completes:
* The interactive 3D Telemetry Dashboard will be hosted at: `http://localhost:3000`
* The Asynchronous Python RL Server will be bound and listening to: `ws://localhost:8000`
```
