# MESA-Earthquake-Rescue-Simulation
Agent Based Modelling


# Earthquake Disaster Response Simulation (Agent-Based Model)

This project is an **Agent-Based Modeling (ABM)** simulation of an urban area affected by an earthquake. It models how **people, buildings, roads, rescue vehicles, drones, and sensors** interact during and after a seismic event.

The simulation focuses on **evacuation dynamics, infrastructure damage, and emergency response behavior**, visualized in real time using the Mesa framework.

---

## Motivation

Earthquake response involves highly dynamic and decentralized decision-making under uncertainty. Static or equation-based models often fail to capture the emergent behavior caused by individual actions, damaged infrastructure, and limited rescue resources.

This project leverages **Agent-Based Modeling** to study how local interactions between agents lead to global outcomes such as evacuation efficiency, congestion, and casualty rates.

---

## Features

* Grid-based urban environment with explicit road networks
* Agent-Based Modeling using the Mesa framework
* Dynamic building damage and collapse simulation
* Temporary road blockages caused by debris
* Civilian evacuation using BFS and A* pathfinding
* Rescue vehicles, drones, and sensors as autonomous agents
* Early warning system via distributed sensors
* Real-time visualization and live metric tracking

---

## Agents Overview

### PersonAgent

* Represents civilians in the city
* States: `normal`, `injured`, `dead`, `evacuated`
* Moves only along road networks
* Evacuates toward the nearest exit
* Injury risk increases near collapsed buildings

### BuildingAgent

* Represents urban infrastructure
* Attributes: health, resilience
* Can transition from healthy → damaged → collapsed
* Collapsed buildings generate debris that blocks nearby roads

### ResourceAgent (Rescue Vehicles)

* Operates on road networks
* Designed to assist injured civilians
* Uses A* pathfinding to navigate dynamically changing roads

### DroneAgent

* Aerial agents not restricted by road blockages
* Used for exploration and situational awareness
* Identifies collapsed buildings over time

### SensorAgent

* Static agents deployed across the city
* Detect earthquake proximity
* Trigger early warning signals with configurable reliability

---

## Environment Design

* City represented as a 2D grid
* Roads explicitly defined for movement
* Buildings placed only on non-road tiles
* Border road cells act as evacuation exits
* Road blockages persist for a limited number of steps

---

## Algorithms Used

* Breadth-First Search (BFS) for evacuation routing
* A* pathfinding with damage-aware traversal cost
* Manhattan distance heuristic
* Time-based debris decay mechanism

---

## Simulation Flow

1. Earthquake event and damage initialization
2. Sensor-based early warning detection
3. Civilian injury and evacuation behavior
4. Dynamic road blockage and clearance
5. Rescue and exploration by autonomous agents
6. Continuous data collection and visualization

---

## Data Collection

The simulation tracks the following metrics over time:

* Number of evacuated civilians
* Number of injured civilians
* Number of fatalities
* Number of collapsed buildings

These metrics are displayed using live charts during runtime.

---

## Technologies Used

* Python 3
* Mesa (Agent-Based Modeling framework)
* Mesa Visualization Server
* Standard Python libraries (random, math, heapq, collections)

---

## Project Structure

```
earthquake_simulation/
├── mesa_earthquake_sim.py
├── README.md
```

---

## How to Run

1. Install dependencies:

   ```
   pip install mesa
   ```

2. Run the simulation:

   ```
   python mesa_earthquake_sim.py
   ```

3. Open the browser at:

   ```
   http://127.0.0.1:8522
   ```

---

## Screenshots
<img width="1919" height="822" alt="Screenshot 2026-01-11 101355" src="https://github.com/user-attachments/assets/4fc9105e-d660-46a5-a917-f05ac8a6d49d" />
<img width="1701" height="1072" alt="Screenshot 2026-01-11 101450" src="https://github.com/user-attachments/assets/81cb5757-bf81-4bbf-9adf-0569b2c9a59c" />
<img width="1421" height="1097" alt="Screenshot 2026-01-11 101523" src="https://github.com/user-attachments/assets/65956307-a062-464c-a92d-141159f8a12a" />


---


## Limitations

* No real-world GIS data integration
* Simplified injury and rescue logic
* No learning or adaptive agent behavior (rule-based)

---

## Future Enhancements

* Reinforcement Learning-based rescue agents
* Real-world map integration
* Multi-aftershock modeling
* Adaptive civilian behavior
* Performance comparison under different response strategies

---

## License

This project is licensed under the MIT License.

---

## Author

Navnita Krishnan
B.Tech Computer Science (AI & Engineering)
* Write a **project abstract**
* Convert this into a **paper-ready methodology section**
