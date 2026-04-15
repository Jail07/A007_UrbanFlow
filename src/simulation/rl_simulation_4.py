import os

import traci
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
SUMO_CFG = "data/routes/scenarios_04/sumo.sumocfg"
LOG_PATH = LOGS_DIR / "urbanflow_detailed_log_04.csv"
RL_LOG_PATH = LOGS_DIR / "rl_training_log_4.csv"

SUMO_BINARY = "sumo-gui"

INTERSECTIONS = {
    "Vefa_244500423": {
        "id": "244500423",
        "in_edges": ["-51095930#0", "622102031#7", "580760138#6", "-620932850#0"],
        "out_edges": ["51095930#1", "-622102031#6", "-580760138#5", "620932850#1"],
        "green_phases": [0, 2],
        "yellow_phases": {0: 1, 2: 3}
    },
    "Kulatov_244500424": {
        "id": "244500424",
        "in_edges": ["-186475564#3", "25684557#5", "186475564#2", "477271462#5"],
        "out_edges": ["186475564#3", "-25684557#5", "-186475564#2", "-477271462#5"],
        "green_phases": [0, 2, 4],
        "yellow_phases": {0: 1, 2: 3, 4: 5}
    },
    "Muka_280015410": {
        "id": "280015410",
        "in_edges": ["1075767203#1", "-477271462#0", None, "192987238#4"],
        "out_edges": ["-1075767203#1", "477271462#0", "49830666#0", "-192987238#4"],
        "green_phases": [0, 2, 4],
        "yellow_phases": {0: 1, 2: 3, 4: 5}
    },
    "Gorko_280015414": {
        "id": "280015414",
        "in_edges": ["49830666#7", "620932850#3", None, "-829563410#0"],
        "out_edges": [None, "-620932850#3", "277523407#0", "829563410#0"],
        "green_phases": [0, 2],
        "yellow_phases": {0: 1, 2: 3}
    }
}



YELLOW_DURATION = 4
MIN_GREEN_TIME = 15

MAX_SIMULATION_STEPS = 86400


class DQN_Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN_Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class HybridPressureAgent:
    def __init__(self, name, state_size=13, action_size=2):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.memory = deque(maxlen=2000)
        self.model = DQN_Network(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.model_path = LOGS_DIR / f"hybrid_brain_{self.name}.pth"

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }, self.model_path)
        print(f"💾 Мозг агента сохранен! (Epsilon: {self.epsilon:.2f})")

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"🧠 Успешная загрузка прошлой памяти! Продолжаем с Epsilon: {self.epsilon:.2f}")
        else:
            print("Память не найдена. Начинаем обучение с чистого листа.")


    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0

        for state, action, reward, next_state in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward + self.gamma * torch.max(self.model(next_state_tensor)[0]).item()
            target_f = self.model(state_tensor)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state_tensor), target_f.detach())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size


def get_hybrid_state(intersection_data):
    in_edges = intersection_data["in_edges"]
    out_edges = intersection_data["out_edges"]
    tls_id = intersection_data["id"]

    state = []
    for edge in in_edges:
        state.append(traci.edge.getLastStepHaltingNumber(edge) if edge else 0)
    for edge in out_edges:
        state.append(traci.edge.getLastStepHaltingNumber(edge) if edge else 0)
    for edge in in_edges:
        state.append(traci.edge.getWaitingTime(edge) if edge else 0)

    state.append(traci.trafficlight.getPhase(tls_id))
    return np.array(state)

def get_hybrid_reward(intersection_data):
    alpha = 1.0
    beta = 0.05
    in_edges = intersection_data["in_edges"]
    out_edges = intersection_data["out_edges"]

    in_queue = sum([traci.edge.getLastStepHaltingNumber(e) for e in in_edges if e])
    out_queue = sum([traci.edge.getLastStepHaltingNumber(e) for e in out_edges if e])
    pressure = abs(in_queue - out_queue)

    total_wait_time = sum([traci.edge.getWaitingTime(e) for e in in_edges if e])

    return - (alpha * pressure + beta * total_wait_time)


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
    return len(buses), "; ".join([f"{b}:[{len(traci.vehicle.getRoute(b))} edges]" for b in buses])

def run_simulation():
    traci.start([SUMO_BINARY, "-c", str(SUMO_CFG)])


    agents = {}
    intersection_states = {}

    for name, data in INTERSECTIONS.items():
        num_actions = len(data["green_phases"])

        agents[name] = HybridPressureAgent(name=name, state_size=13, action_size=num_actions)

        try:
            agents[name].load_model()
        except Exception as e:

            print(f"[{name}] Старая память не подошла или отсутствует. Начинаем с нуля.")

        intersection_states[name] = {
            "last_switch_time": 0,
            "target_phase": None,
            "current_state": None,
            "last_action": None
        }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    f_traf = open(str(LOG_PATH), "w", newline="")
    f_rl = open(str(RL_LOG_PATH), "w", newline="")
    writer_traf = csv.writer(f_traf)
    writer_rl = csv.writer(f_rl)

    writer_traf.writerow(
        ["time", "intersection", "queue", "vehicles", "mean_speed", "lane_count", "lane_configs", "bus_count",
         "bus_routes"])
    writer_rl.writerow(["step", "agent", "reward", "loss", "epsilon"])

    print("Симуляция Multi-Agent Hybrid RL запущена...")
    step_counter = 0
    t = 0

    valid_green_phases = {}
    tl_ids = traci.trafficlight.getIDList()

    for tls_id in tl_ids:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        green_phases = [i for i, phase in enumerate(logic.phases) if 'G' in phase.state or 'g' in phase.state]
        valid_green_phases[tls_id] = green_phases

    while traci.simulation.getMinExpectedNumber() > 0 and t < MAX_SIMULATION_STEPS:
        traci.simulationStep()
        t = int(traci.simulation.getTime())

        for name, data in INTERSECTIONS.items():
            agent = agents[name]
            st = intersection_states[name]
            tls_id = data["id"]

            green_phases = data["green_phases"]
            yellow_dict = data["yellow_phases"]

            current_phase = traci.trafficlight.getPhase(tls_id)

            if current_phase in yellow_dict.values():
                if t - st["last_switch_time"] >= YELLOW_DURATION:
                    if st["target_phase"] is not None:
                        next_green = st["target_phase"]
                        traci.trafficlight.setPhase(tls_id, next_green)
                        traci.trafficlight.setPhaseDuration(tls_id, 1000)
                        st["target_phase"] = None
                    else:
                        num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
                        next_green = (current_phase + 1) % num_phases
                        traci.trafficlight.setPhase(tls_id, next_green)
                        traci.trafficlight.setPhaseDuration(tls_id, 1000)

                    st["last_switch_time"] = t

            elif current_phase in green_phases:
                if t - st["last_switch_time"] >= MIN_GREEN_TIME:

                    if st["last_action"] is not None:
                        reward = get_hybrid_reward(data)
                        next_state = get_hybrid_state(data)
                        agent.remember(st["current_state"], st["last_action"], reward, next_state)
                        loss = agent.replay(batch_size=32)

                        step_counter += 1
                        loss_val = loss if loss is not None else 0
                        writer_rl.writerow([step_counter, name, reward, loss_val, agent.epsilon])

                    st["current_state"] = get_hybrid_state(data)
                    action_index = agent.act(st["current_state"])
                    st["last_action"] = action_index

                    if action_index < len(green_phases):
                        desired_phase = green_phases[action_index]
                    else:
                        desired_phase = green_phases[0]

                    if desired_phase != current_phase:
                        st["target_phase"] = desired_phase
                        yellow_phase = yellow_dict[current_phase]
                        traci.trafficlight.setPhase(tls_id, yellow_phase)
                        traci.trafficlight.setPhaseDuration(tls_id, YELLOW_DURATION)
                        st["last_switch_time"] = t
                        print(
                            f"[{datetime.timedelta(seconds=t)}] ИИ ({name}) переключает на фазу {desired_phase}. Epsilon: {agent.epsilon:.3f}")
                    else:
                        traci.trafficlight.setPhase(tls_id, current_phase)
                        traci.trafficlight.setPhaseDuration(tls_id, 1000)
                        st["last_switch_time"] = t
                        print(f"[{datetime.timedelta(seconds=t)}] ИИ ({name}) продлевает фазу {current_phase}.")

        if t % 10 == 0:
            for name, data in INTERSECTIONS.items():
                in_edges = data["in_edges"]
                q_total, v_total, s_sum, total_lanes = 0, 0, 0, 0
                all_lane_configs = []

                for e in in_edges:
                    if e == None:
                        continue
                    q_total += traci.edge.getLastStepHaltingNumber(e)
                    v_total += traci.edge.getLastStepVehicleNumber(e)
                    s_sum += traci.edge.getLastStepMeanSpeed(e)
                    n_lanes, l_cfg = get_lane_info(e)
                    total_lanes += n_lanes
                    all_lane_configs.append(l_cfg)

                avg_speed = round((s_sum / len(in_edges)) * 3.6, 2) if v_total > 0 else 0
                writer_traf.writerow(
                    [t, name, q_total, v_total, avg_speed, total_lanes, " / ".join(all_lane_configs), 0, ""])

    f_traf.close()
    f_rl.close()

    for name, agent in agents.items():
        agent.save_model()

    traci.close()
    print("Симуляция завершена. Логи и мозги сохранены.")


def analyze_results():
    try:
        df_traf = pd.read_csv(str(LOG_PATH))
        df_rl = pd.read_csv(str(RL_LOG_PATH))

        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        fig.suptitle("Отчет об обучении: 4 Перекрестка", fontsize=16, fontweight='bold')

        colors = plt.cm.tab10.colors
        agents = sorted(df_rl['agent'].unique())

        axes[0].set_title("1. Суммарные очереди (Сглаженный тренд MA 10)")
        pivot_q = df_traf.pivot_table(index="time", columns="intersection", values="queue", aggfunc='sum')

        pivot_q_smoothed = pivot_q.rolling(window=10, min_periods=1).mean()

        for idx, col in enumerate(sorted(pivot_q_smoothed.columns)):
            axes[0].plot(pivot_q_smoothed.index, pivot_q_smoothed[col],
                         linewidth=2, label=col, color=colors[idx % len(colors)], alpha=0.9)

        axes[0].set_ylabel("Кол-во машин")
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Выносим легенду вправо

        axes[1].set_title("2. Прогресс обучения: Награда (Сглаженный тренд MA 20)")

        for idx, agent_name in enumerate(agents):
            agent_data = df_rl[df_rl['agent'] == agent_name].copy()
            agent_data['reward_smoothed'] = agent_data['reward'].rolling(window=20, min_periods=1).mean()
            axes[1].plot(agent_data['step'], agent_data['reward_smoothed'],
                         linewidth=2, label=agent_name, color=colors[idx % len(colors)])

        axes[1].set_ylabel("Награда (Ближе к 0 = Лучше)")
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        axes[2].set_title("3. Исследование среды (Уровень случайности)")

        for idx, agent_name in enumerate(agents):
            agent_data = df_rl[df_rl['agent'] == agent_name]
            axes[2].plot(agent_data['step'], agent_data['epsilon'],
                         linewidth=2, label=agent_name, color=colors[idx % len(colors)])

        axes[2].set_ylabel("Эпсилон")
        axes[2].set_xlabel("Шаги принятия решений")
        axes[2].grid(True, linestyle='--', alpha=0.7)
        axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
        plt.show()

    except Exception as e:
        print(f"Ошибка при анализе логов: {e}")

if __name__ == "__main__":
    try:
        run_simulation()
        analyze_results()
    except KeyboardInterrupt:
        print("Прервано пользователем")
        traci.close()
        analyze_results()
    except traci.exceptions.FatalTraCIError as e:
        print(f"TraCI Error: {e}")
        traci.close()