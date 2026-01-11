import random
import math
import heapq
import itertools
from collections import deque

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter


# -------------------------
# UI legend (TextElement)
# -------------------------
class Legend(TextElement):
    # Render sidebar legend and simple rescue-efficiency metric.
    def render(self, model):
        # Count totals to compute efficiency percentage.
        total_people = model.population
        evacuated = len([a for a in model.schedule.agents if isinstance(a, PersonAgent) and a.state == "evacuated"])
        dead = len([a for a in model.schedule.agents if isinstance(a, PersonAgent) and a.state == "dead"])
        injured = len([a for a in model.schedule.agents if isinstance(a, PersonAgent) and a.state == "injured"])
        
        efficiency = 0
        if total_people > 0:
            efficiency = int((evacuated / total_people) * 100)
        
        # Map efficiency to a grade and color for display.
        if efficiency >= 90:
            grade = "A+ 🌟"
            color = "#00cc00"
        elif efficiency >= 75:
            grade = "A 👍"
            color = "#33cc33"
        elif efficiency >= 60:
            grade = "B ✓"
            color = "#66cc66"
        elif efficiency >= 40:
            grade = "C ⚠️"
            color = "#ff9900"
        else:
            grade = "D ⛔"
            color = "#cc0000"
        
        # Return HTML string shown in the UI.
        html = f"""
        <div style="font-family: Arial; line-height:1.6; padding-left:10px;">
          <h3>Legend</h3>
          <div>🏢 &nbsp; Building (healthy)</div>
          <div>🏚️ &nbsp; Building (damaged)</div>
          <div>🧱💥 &nbsp; Collapsed building</div>
          <div style="margin-top:8px;">👤 &nbsp; Person (healthy)</div>
          <div>🆘 &nbsp; Person (injured)</div>
          <div>💀 &nbsp; Person (dead)</div>
          <div>✅ &nbsp; Person (evacuated)</div>
          <div style="margin-top:8px;">🚑 &nbsp; Rescue vehicle</div>
          <div>🚁 &nbsp; Drone</div>
          <div>🛰️ &nbsp; Sensor (broadcasts early warning)</div>
          <div style="margin-top:12px; padding:10px; background:#f0f0f0; border-radius:5px;">
            <strong>Rescue Efficiency</strong><br/>
            <span style="font-size:20px; color:{color}; font-weight:bold;">{efficiency}%</span> 
            <span style="font-size:18px;">{grade}</span><br/>
            <small>Evacuated: {evacuated}/{total_people} | Dead: {dead} | Injured: {injured}</small>
          </div>
        </div>
        """
        return html


# -------------------------
# Agents
# -------------------------
class RoadAgent(Agent):
    """Visual placeholder for road tiles (walkable)."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class BuildingAgent(Agent):
    # Building has resilience and health; collapse state updated by quake_impact.
    def __init__(self, unique_id, model, resilience=0.7):
        super().__init__(unique_id, model)
        self.resilience = resilience
        self.health = 1.0
        self.collapsed = False
        self.discovered = False

    # Apply quake damage to this building; may change collapsed flag.
    def quake_impact(self, intensity):
        # Damage scales with intensity and inverse resilience, with random multiplier.
        damage = intensity * (1 - self.resilience) * random.uniform(1.5, 2.2)
        self.health = max(0.0, self.health - damage)
        # If below threshold, mark as collapsed.
        if self.health <= 0.45 and not self.collapsed:
            self.collapsed = True

    def step(self):
        # Building has no per-tick behavior beyond state.
        pass


class PersonAgent(Agent):
    # Person can be normal, injured, dead, or evacuated. Moves toward nearest exit.
    def __init__(self, unique_id, model, speed=1):
        super().__init__(unique_id, model)
        self.state = "normal"
        self.base_speed = speed
        self.speed = speed
        self.path = None
        self.target = None
        self.discovered = False  # drones mark injured people as discovered

    # Plan path to the nearest exit if no target exists.
    def plan(self):
        if self.target is None:
            self.target = self.model.find_nearest_exit(self.pos)
        if self.target is not None:
            # BFS path is used for simple pedestrian routing.
            self.path = self.model.find_path_bfs(self.pos, self.target)
            if self.path is None:
                self.path = []

    def step(self):
        # If injured / dead / evacuated, handle special state transitions.
        if self.state in ("injured", "dead", "evacuated"):
            # Injured people have a small chance to die over time.
            if self.state == "injured" and hasattr(self, "injury_time"):
                t = max(0, self.model.schedule.time - self.injury_time)
                if random.random() < 0.003 * t:
                    self.state = "dead"
            # Injured/dead/evacuated do not move further.
            return

        # Normal movement: plan if needed and follow path one or more steps.
        self.speed = max(1, int(self.base_speed))
        if self.path is None:
            self.plan()

        if self.path:
            # Step through path up to speed, aborting if next tile isn't road.
            steps = min(self.speed, len(self.path))
            for _ in range(steps):
                if not self.path:
                    break
                nxt = self.path.pop(0)
                # If the next tile is not road, abandon current path and replan later.
                if not self.model.is_road(nxt):
                    self.path = None
                    break
                try:
                    self.model.grid.move_agent(self, nxt)
                except Exception:
                    # If move fails, drop the path and retry next tick.
                    self.path = None
                    break

        # If on an exit cell, mark as evacuated.
        if self.pos in self.model.exit_cells:
            self.state = "evacuated"


class ResourceAgent(Agent):
    # Rescue vehicle that searches for injured people and transports them to exits.
    def __init__(self, unique_id, model, speed=2):
        super().__init__(unique_id, model)
        self.speed = speed
        self.target = None
        self.carrying = []

    # Return nearest injured, prioritizing those discovered by drones.
    def find_nearest_injured(self):
        injured = [a for a in self.model.schedule.agents if isinstance(a, PersonAgent) and a.state == "injured"]
        if not injured:
            return None
        # Sort so discovered injured come first, then by manhattan distance.
        injured.sort(key=lambda p: (0 if getattr(p, "discovered", False) else 1,
                                    self.model.manhattan_distance(self.pos, p.pos)))
        return injured[0]

    # Move along a precomputed path up to speed steps.
    def move_along(self, path):
        if not path:
            return
        steps = min(self.speed, len(path))
        for _ in range(steps):
            if not path:
                break
            nxt = path.pop(0)
            if not self.model.is_road(nxt):
                break
            try:
                self.model.grid.move_agent(self, nxt)
            except Exception:
                break

    def step(self):
        # If carrying someone, drive toward nearest exit and drop them off there.
        if self.carrying:
            exit_pos = self.model.find_nearest_exit(self.pos)
            if exit_pos:
                path = self.model.find_path_astar(self.pos, exit_pos)
                self.move_along(path or [])
            # If at exit, set carried persons to evacuated state and clear cargo.
            if self.pos in self.model.exit_cells:
                for p in self.carrying:
                    p.state = "evacuated"
                    p.discovered = False
                self.carrying = []
            return

        # Otherwise, look for injured to rescue.
        if self.target is None or self.target.state != "injured":
            self.target = self.find_nearest_injured()
        if self.target is None:
            # No injured found: wander to a random road neighbor.
            neigh = self.model.road_neighbors(self.pos)
            random.shuffle(neigh)
            for n in neigh:
                try:
                    self.model.grid.move_agent(self, n)
                except Exception:
                    pass
                break
            return

        # Move toward target using A*; pick up if reached.
        path = self.model.find_path_astar(self.pos, self.target.pos)
        self.move_along(path or [])
        if self.pos == self.target.pos and len(self.carrying) < 1:
            # Stabilize carried person (set normal) and pick them up.
            self.target.state = "normal"
            self.target.discovered = False
            if hasattr(self.target, "injury_time"):
                try:
                    delattr(self.target, "injury_time")
                except Exception:
                    pass
            self.carrying.append(self.target)
            self.target = None


class DroneAgent(Agent):
    # Drone flies and discovers collapsed buildings and injured people nearby.
    def __init__(self, unique_id, model, speed=3):
        super().__init__(unique_id, model)
        self.speed = speed
        self.visited = set()

    def step(self):
        # Choose a neighbor cell to move to; prefer cells near undiscovered collapsed buildings.
        neigh = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        random.shuffle(neigh)
        chosen = None
        for n in neigh:
            for a in self.model.grid.get_cell_list_contents([n]):
                if isinstance(a, BuildingAgent) and a.collapsed and not a.discovered:
                    chosen = n
                    break
            if chosen:
                break
        # If none with undiscovered collapse, prefer unvisited neighbors.
        if chosen is None:
            neigh_sorted = sorted(neigh, key=lambda p: (p in self.visited))
            if neigh_sorted:
                chosen = neigh_sorted[0]
        # Move if a choice exists.
        if chosen:
            try:
                self.model.grid.move_agent(self, chosen)
            except Exception:
                pass
            # On arrival, mark collapsed buildings discovered and mark nearby injured as discovered.
            for a in self.model.grid.get_cell_list_contents([self.pos]):
                if isinstance(a, BuildingAgent) and a.collapsed:
                    a.discovered = True
                    # Scan radius 2 around drone; mark injured people discovered for ambulances.
                    scan = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True, radius=2)
                    for s in scan:
                        for obj in self.model.grid.get_cell_list_contents([s]):
                            if isinstance(obj, PersonAgent) and obj.state == "injured":
                                obj.discovered = True
            # Record visited position to bias future movement.
            self.visited.add(self.pos)


class SensorAgent(Agent):
    # Simple sensor for early warning broadcasting.
    def __init__(self, unique_id, model, coverage=5, reliability=0.9):
        super().__init__(unique_id, model)
        self.coverage = coverage
        self.reliability = reliability
        self.active = False

    # If within coverage of epicenter and passes reliability roll, activate model-wide early warning.
    def detect_and_broadcast(self, epicenter, magnitude):
        if self.model.manhattan_distance(self.pos, epicenter) <= self.coverage and random.random() < self.reliability:
            self.active = True
            self.model.early_warning_active = True

    def step(self):
        # Sensors are passive after initial detection.
        pass


# -------------------------
# Model
# -------------------------
class EarthquakeModel(Model):
    # Model initializes grid, agents, and applies initial quake damage.
    def __init__(self, width=24, height=16, population=30,
                 resources=2, sensors=2, drones=1, magnitude=0.7):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        # RandomActivation gives each agent a chance to act in random order each step.
        self.schedule = RandomActivation(self)
        self.population = population
        self.resources = resources
        self.sensors = sensors
        self.drones = drones
        self.magnitude = magnitude

        # Build road map: every 3rd row or every 4th column is road for balanced coverage.
        self.road_map = set()
        for x in range(self.width):
            for y in range(self.height):
                if (y % 3 == 0) or (x % 4 == 0):
                    self.road_map.add((x, y))

        # Place RoadAgent objects on each road tile for visualization (layer 0).
        rid = 0
        for pos in list(self.road_map):
            r = RoadAgent(f"road_{rid}", self)
            self.schedule.add(r)
            self.grid.place_agent(r, pos)
            rid += 1

        # Place buildings on non-road tiles with specified density.
        self.building_list = []
        bid = 0
        building_density = 0.70
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) in self.road_map:
                    continue
                # Each non-road tile becomes a building with probability building_density.
                if random.random() < building_density:
                    b = BuildingAgent(f"b_{bid}", self, resilience=random.uniform(0.30, 0.70))
                    self.schedule.add(b)
                    self.grid.place_agent(b, (x, y))
                    self.building_list.append(b)
                    bid += 1

        # Choose an epicenter near center and apply quake impact to all buildings.
        self.epicenter = (self.width // 2 + random.randint(-2, 2), self.height // 2 + random.randint(-2, 2))
        maxdist = math.hypot(self.width, self.height)
        for b in self.building_list:
            # Damage intensity falls off with distance from epicenter.
            dist = self.manhattan_distance(b.pos, self.epicenter)
            intensity = self.magnitude * max(0, 1 - (dist / maxdist) * 1.5)
            b.quake_impact(intensity)

        # Identify exit cells as road cells on the map border for evacuation targets.
        self.exit_cells = [pos for pos in self.road_map if pos[0] in (0, self.width - 1) or pos[1] in (0, self.height - 1)]
        if not self.exit_cells:
            # Fallback: use a few road cells if border roads are absent.
            self.exit_cells = list(self.road_map)[:4]

        # Place people randomly on road cells, up to population limit.
        pid = 0
        road_positions = list(self.road_map)
        random.shuffle(road_positions)
        placed = 0
        for pos in road_positions:
            if placed >= self.population:
                break
            occupants = [a for a in self.grid.get_cell_list_contents([pos]) if isinstance(a, PersonAgent)]
            if len(occupants) >= 2:
                # Limit density: max 2 people per road tile.
                continue
            p = PersonAgent(f"p_{pid}", self, speed=random.randint(1, 1))
            self.schedule.add(p)
            self.grid.place_agent(p, pos)
            pid += 1
            placed += 1

        # Place rescue vehicles at random border exit roads.
        rid = 0
        border_roads = [r for r in self.exit_cells]
        for i in range(self.resources):
            pos = random.choice(border_roads)
            rv = ResourceAgent(f"res_{rid}", self, speed=2)
            self.schedule.add(rv)
            self.grid.place_agent(rv, pos)
            rid += 1

        # Place sensors on random road cells to attempt early warning broadcast.
        sid = 0
        for _ in range(self.sensors):
            pos = random.choice(road_positions)
            s = SensorAgent(f"s_{sid}", self, coverage=5, reliability=0.95)
            self.schedule.add(s)
            self.grid.place_agent(s, pos)
            sid += 1

        # Place drones at center start position (can be tuned).
        did = 0
        for _ in range(self.drones):
            pos = (self.width // 2, self.height // 2)
            d = DroneAgent(f"d_{did}", self, speed=3)
            self.schedule.add(d)
            self.grid.place_agent(d, pos)
            did += 1

        # Data collector records model-level stats each tick.
        self.datacollector = DataCollector(
            model_reporters={
                "Evacuated": lambda m: len([a for a in m.schedule.agents if isinstance(a, PersonAgent) and a.state == "evacuated"]),
                "Injured": lambda m: len([a for a in m.schedule.agents if isinstance(a, PersonAgent) and a.state == "injured"]),
                "Dead": lambda m: len([a for a in m.schedule.agents if isinstance(a, PersonAgent) and a.state == "dead"]),
                "CollapsedBuildings": lambda m: len([b for b in m.building_list if b.collapsed]),
            }
        )

        # Sensors attempt to broadcast early warning immediately after placement.
        for a in list(self.schedule.agents):
            if isinstance(a, SensorAgent):
                a.detect_and_broadcast(self.epicenter, self.magnitude)

        # Injure people who are near collapsed buildings as initial effect.
        self._injure_people_near_collapses()

        self.running = True
        self.early_warning_active = False

    # -------------------------
    # Model step & helpers
    # -------------------------
    def step(self):
        # Stop condition: if no active people remain (all evacuated or dead), stop simulation.
        people = [a for a in self.schedule.agents if isinstance(a, PersonAgent)]
        if people:
            active_people = [p for p in people if p.state not in ("evacuated", "dead")]
            if not active_people:
                # All people processed; print summary and halt.
                self.running = False
                print(f"\n=== SIMULATION COMPLETE at step {self.schedule.time} ===")
                evacuated = len([p for p in people if p.state == "evacuated"])
                dead = len([p for p in people if p.state == "dead"])
                print(f"Evacuated: {evacuated}, Dead: {dead}, Total: {len(people)}")
                return

        # Periodic aftershock: pick a damaged building occasionally and apply more impact.
        if self.schedule.time > 0 and self.schedule.time % 10 == 0:
            candidates = [b for b in self.building_list if not b.collapsed and b.health < 0.85]
            if candidates:
                b = random.choice(candidates)
                b.quake_impact(self.magnitude * 0.75)

        # Collect data for charts, then advance all agents one tick.
        self.datacollector.collect(self)
        self.schedule.step()

        # Reset early warning after 30 ticks to avoid permanent active state.
        if self.schedule.time > 30:
            self.early_warning_active = False

    # Helper: is this position a road?
    def is_road(self, pos):
        return pos in self.road_map

    # Return adjacent road neighbors for pathfinding/movement.
    def road_neighbors(self, pos):
        candidates = self.grid.get_neighborhood(pos, moore=False, include_center=False)
        return [p for p in candidates if self.is_road(p)]

    # Manhattan distance used as heuristic and simple proximity measures.
    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Find nearest exit cell (used as evacuation target).
    def find_nearest_exit(self, pos):
        if not self.exit_cells:
            return None
        return min(self.exit_cells, key=lambda e: self.manhattan_distance(pos, e))

    # BFS pathfinder over road graph (unweighted).
    def find_path_bfs(self, start, goal):
        if start == goal:
            return []
        q = deque([(start, [])])
        visited = {start}
        while q:
            curr, path = q.popleft()
            for n in self.road_neighbors(curr):
                if n in visited:
                    continue
                newp = path + [n]
                if n == goal:
                    return newp
                visited.add(n)
                q.append((n, newp))
        return None

    # Traversal cost considers nearby damaged buildings to penalize risky routes.
    def traversal_cost(self, pos):
        penalty = 0.0
        for c in self.grid.get_neighborhood(pos, moore=True, include_center=False):
            for a in self.grid.get_cell_list_contents([c]):
                if isinstance(a, BuildingAgent) and a.health < 0.6:
                    penalty += (1.0 - a.health) * 0.5
        return 1.0 + penalty

    # A* pathfinder that uses manhattan heuristic and traversal_cost as movement cost.
    def find_path_astar(self, start, goal):
        if start == goal:
            return []
        frontier = []
        counter = itertools.count()
        heapq.heappush(frontier, (self.manhattan_distance(start, goal), next(counter), 0.0, start, []))
        best_g = {start: 0.0}
        while frontier:
            f, _cnt, g, curr, path = heapq.heappop(frontier)
            if curr == goal:
                return path
            for n in self.road_neighbors(curr):
                cost = self.traversal_cost(n)
                newg = g + cost
                if n not in best_g or newg < best_g[n]:
                    best_g[n] = newg
                    h = self.manhattan_distance(n, goal)
                    heapq.heappush(frontier, (newg + h, next(counter), newg, n, path + [n]))
        return None

    # Injure people located near collapsed buildings (scan radius and probabilistic).
    def _injure_people_near_collapses(self):
        radius = 2
        for b in [a for a in self.building_list if a.collapsed]:
            # Scan neighborhood around collapsed building.
            scan = self.grid.get_neighborhood(b.pos, moore=True, include_center=True, radius=radius)
            for pos in scan:
                dist = self.manhattan_distance(b.pos, pos)
                # Closer positions have higher weight for injury chance.
                weight = max(0.0, 1.0 - (dist / (radius + 0.1)))
                for p in self.grid.get_cell_list_contents([pos]):
                    if isinstance(p, PersonAgent) and p.state == "normal":
                        base = 0.30  # base injury probability
                        damage_factor = (1 - b.health) * 0.75
                        prob = min(0.95, base + damage_factor * weight + random.uniform(0, 0.18))
                        if random.random() < prob:
                            p.state = "injured"
                            p.injury_time = self.schedule.time
                            p.discovered = False  # drones required to mark discovered


# -------------------------
# Portrayal
# -------------------------
def portrayal(agent):
    # Map each agent type to a visualization dictionary used by CanvasGrid.
    if agent is None:
        return None
    if isinstance(agent, RoadAgent):
        return {"Shape": "rect", "w": 1, "h": 1, "Filled": True, "Layer": 0, "Color": "#dcdcdc"}
    if isinstance(agent, BuildingAgent):
        # Different appearance depending on collapse and health.
        if agent.collapsed:
            return {"Shape": "rect", "w": 1, "h": 1, "Filled": True, "Layer": 1, "Color": "#8b0000",
                    "text": "🧱💥", "text_color": "white"}
        elif agent.health > 0.7:
            return {"Shape": "rect", "w": 1, "h": 1, "Filled": True, "Layer": 1, "Color": "#7cfc00",
                    "text": "🏢", "text_color": "black"}
        else:
            return {"Shape": "rect", "w": 1, "h": 1, "Filled": True, "Layer": 1, "Color": "#ff8c00",
                    "text": "🏚️", "text_color": "black"}
    if isinstance(agent, PersonAgent):
        # Show different icons/colors for person states and mark discovered injured specially.
        state_map = {
            "normal": ("👤", "blue"),
            "injured": ("🆘", "red"),
            "dead": ("💀", "black"),
            "evacuated": ("✅", "green")
        }
        txt, color = state_map.get(agent.state, ("👤", "blue"))
        text_display = txt
        if agent.state == "injured" and getattr(agent, "discovered", False):
            text_display = "🚨"
        return {"Shape": "circle", "r": 0.4, "Filled": True, "Layer": 3, "Color": color, "text": text_display, "text_color": "white"}
    if isinstance(agent, ResourceAgent):
        return {"Shape": "rect", "w": 0.6, "h": 0.6, "Filled": True, "Layer": 4, "Color": "#003399", "text": "🚑"}
    if isinstance(agent, DroneAgent):
        return {"Shape": "rect", "w": 0.45, "h": 0.45, "Filled": True, "Layer": 5, "Color": "gold", "text": "🚁"}
    if isinstance(agent, SensorAgent):
        # Sensor color reflects whether early warning was activated.
        color = "cyan" if agent.active or agent.model.early_warning_active else "lightblue"
        return {"Shape": "circle", "r": 0.25, "Filled": True, "Layer": 2, "Color": color, "text": "🛰️"}
    return None


# -------------------------
# Server & Launch
# -------------------------
# Create grid and chart UI modules; ensure data_collector_name matches model attribute.
grid = CanvasGrid(portrayal, 24, 16, 700, 500)
chart = ChartModule(
    [{"Label": "Evacuated", "Color": "Green"},
     {"Label": "Injured", "Color": "Orange"},
     {"Label": "Dead", "Color": "Black"},
     {"Label": "CollapsedBuildings", "Color": "Red"}],
    data_collector_name='datacollector'
)
legend = Legend()

# ModularServer exposes the model + visualization in a web UI.
server = ModularServer(
    EarthquakeModel,
    [legend, grid, chart],
    "Earthquake Rescue Sim with Efficiency Grading 🌟",
    {
        "width": 24,
        "height": 16,
        "population": 30,
        "resources": UserSettableParameter("slider", "Rescue vehicles", 2, 0, 5, 1),
        "sensors": 2,
        "drones": UserSettableParameter("slider", "Drones", 1, 0, 5, 1),
        "magnitude": 0.7,
    }
)

if __name__ == "__main__":
    # Use a consistent port number in both server.port and printed message.
    server.port = 8522
    print("Launching server at http://127.0.0.1:8523")
    server.launch()