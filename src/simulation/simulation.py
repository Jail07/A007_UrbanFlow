import traci
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
SUMO_CFG = "data/routes/scenarios_03/sumo.sumocfg"
LOG_PATH = LOGS_DIR / "urbanflow_detailed_log_02.csv"

SUMO_BINARY = "sumo-gui"
# SUMO_BINARY = "sumo"
TLS_ID = "244500423"


EDGES = {
    "W": ["-622102031#6", "622102031#6"],
    "N": ["-51095930#1", "51095930#1"],
    "E": ["-620932850#1", "620932850#1"],
    "S": ["-580760138#5", "580760138#5"]
}

IN_EDGES = {
    "W": "622102031#6",
    "N": "51095930#1",
    "E": "620932850#1",
    "S": "580760138#5",
}
OUT_EDGES = {
    "W": "-622102031#6",
    "N": "-51095930#1",
    "E": "-620932850#1",
    "S": "-580760138#5",
}

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2
PHASE_EW_YELLOW = 3

YELLOW_DURATION = 4
MIN_GREEN_TIME = 15


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PressLightAgent:
    def __init__(self, state_size=9, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward + self.gamma * torch.max(self.model(next_state_tensor)[0]).item()
            target_f = self.model(state_tensor)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_tensor), target_f.detach())
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def get_edge_queue(edge_id):
    return traci.edge.getLastStepHaltingNumber(edge_id)


def get_current_state():
    state = [
        get_edge_queue(IN_EDGES["N"]), get_edge_queue(IN_EDGES["S"]),
        get_edge_queue(IN_EDGES["E"]), get_edge_queue(IN_EDGES["W"]),
        get_edge_queue(OUT_EDGES["N"]), get_edge_queue(OUT_EDGES["S"]),
        get_edge_queue(OUT_EDGES["E"]), get_edge_queue(OUT_EDGES["W"]),
        traci.trafficlight.getPhase(TLS_ID)
    ]
    return np.array(state)


def calculate_pressures():
    q_in_ns = get_edge_queue(IN_EDGES["N"]) + get_edge_queue(IN_EDGES["S"])
    q_out_ns = get_edge_queue(OUT_EDGES["N"]) + get_edge_queue(OUT_EDGES["S"])

    q_in_ew = get_edge_queue(IN_EDGES["E"]) + get_edge_queue(IN_EDGES["W"])
    q_out_ew = get_edge_queue(OUT_EDGES["E"]) + get_edge_queue(OUT_EDGES["W"])

    pressure_ns = q_in_ns - q_out_ns
    pressure_ew = q_in_ew - q_out_ew
    return pressure_ns, pressure_ew


def get_reward():
    pressure_ns, pressure_ew = calculate_pressures()
    return -(abs(pressure_ns) + abs(pressure_ew))


def get_lane_info(edge_id):
    num_lanes = traci.edge.getLaneNumber(edge_id)
    lane_data = []
    for i in range(num_lanes):
        lane_id = f"{edge_id}_{i}"
        links = traci.lane.getLinks(lane_id)
        directions = "".join(sorted(set(link[6] for link in links)))
        lane_data.append(f"L{i}({directions})")
    return num_lanes, "|".join(lane_data)


def get_buses_on_edge(edge_id):
    vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
    buses = [v for v in vehicles if traci.vehicle.getVehicleClass(v) == "bus"]
    bus_routes = [f"{b}:[{len(traci.vehicle.getRoute(b))} edges]" for b in buses]
    return len(buses), "; ".join(bus_routes)


def run_simulation():
    traci.start([SUMO_BINARY, "-c", str(SUMO_CFG)])

    agent = PressLightAgent(state_size=9, action_size=2)
    last_action_time = 0

    current_state = None
    last_action = None
    target_phase = None

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(str(LOG_PATH), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "dir", "queue", "vehicles", "mean_speed",
            "lane_count", "lane_configs", "bus_count", "bus_routes"
        ])

        print("Симуляция PressLight запущена...")

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            t = int(traci.simulation.getTime())
            current_phase = traci.trafficlight.getPhase(TLS_ID)


            if current_phase in [PHASE_NS_YELLOW, PHASE_EW_YELLOW]:
                if t - last_action_time >= YELLOW_DURATION:
                    traci.trafficlight.setPhase(TLS_ID, target_phase)
                    last_action_time = t


            elif t - last_action_time >= MIN_GREEN_TIME and current_phase in [PHASE_NS_GREEN, PHASE_EW_GREEN]:


                if last_action is not None:
                    reward = get_reward()
                    next_state = get_current_state()
                    agent.remember(current_state, last_action, reward, next_state)
                    agent.replay()


                current_state = get_current_state()
                action = agent.get_action(current_state)
                last_action = action

                desired_green_phase = PHASE_NS_GREEN if action == 0 else PHASE_EW_GREEN

                if desired_green_phase != current_phase:
                    target_phase = desired_green_phase
                    yellow_phase = PHASE_NS_YELLOW if current_phase == PHASE_NS_GREEN else PHASE_EW_YELLOW
                    traci.trafficlight.setPhase(TLS_ID, yellow_phase)
                    last_action_time = t
                    print(f"[{t}c] Агент сменил фазу. Epsilon (Шанс random): {agent.epsilon:.2f}")
                else:

                    last_action_time = t


            for direction, edges in EDGES.items():
                q_total, v_total, s_sum, b_total, total_lanes = 0, 0, 0, 0, 0
                all_lane_configs, all_bus_routes = [], []

                for e in edges:
                    q_total += traci.edge.getLastStepHaltingNumber(e)
                    v_total += traci.edge.getLastStepVehicleNumber(e)
                    s_sum += traci.edge.getLastStepMeanSpeed(e)

                    n_lanes, l_cfg = get_lane_info(e)
                    total_lanes += n_lanes
                    all_lane_configs.append(l_cfg)

                    b_count, b_rts = get_buses_on_edge(e)
                    b_total += b_count
                    if b_rts: all_bus_routes.append(b_rts)

                avg_speed = round((s_sum / len(edges)) * 3.6, 2) if v_total > 0 else 0

                writer.writerow([
                    t, direction, q_total, v_total, avg_speed,
                    total_lanes, " / ".join(all_lane_configs),
                    b_total, " | ".join(all_bus_routes)
                ])

    traci.close()
    print(f"Данные сохранены в {LOG_PATH}")


    torch.save(agent.model.state_dict(), "presslight_model.pth")



def analyze_results():
    df = pd.read_csv(str(LOG_PATH))
    pivot_q = df.pivot(index="time", columns="dir", values="queue")
    pivot_q.plot(figsize=(10, 5))
    plt.title("Динамика очередей (PressLight)")
    plt.ylabel("Количество стоящих машин")
    plt.xlabel("Время симуляции (сек)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        run_simulation()
        analyze_results()
    except KeyboardInterrupt:
        print("\nСимуляция прервана пользователем")
        traci.close()
    except traci.exceptions.FatalTraCIError as e:
        print(f"Ошибка TraCI: {e}")
        traci.close()