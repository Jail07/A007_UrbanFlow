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
LOG_PATH = LOGS_DIR / "urbanflow_detailed_log.csv"
RL_LOG_PATH = LOGS_DIR / "rl_training_log.csv"

SUMO_BINARY = "sumo-gui"

INTERSECTIONS = {
    "Vefa_244500423": {
        "id": "244500423",
        "in_edges": ["-51095930#1", "622102031#6", "-620932850#1", "580760138#5"],
        "out_edges": ["51095930#1", "-622102031#6", "620932850#1", "-580760138#5"],
        "green_phases": [0, 2],           # ИИ выбирает между 0 и 2
        "yellow_phases": {0: 1, 2: 3}     # Уходим с 0 -> желтый 1. Уходим с 2 -> желтый 3.
    },
    "Kulatov_244500424": {
        "id": "244500424",
        "in_edges": ["-186475564#4", "25684557#4", "186475564#1", "477271462#4"],
        "out_edges": ["186475564#4", "-25684557#4", "-186475564#1", "-477271462#4"],
        "green_phases": [0, 2, 4],        # У ИИ теперь 3 варианта действий (Action size = 3)
        "yellow_phases": {0: 1, 2: 3, 4: 5}
    },
    "Muka_280015410": {
        "id": "280015410",
        "in_edges": ["1075767203#0", "-477271462#1", None, "192987238#2"],
        "out_edges": ["-1075767203#0", "477271462#1", "49830666#1", "-192987238#2"],
        "green_phases": [0, 2, 4],        # Тоже 3 варианта
        "yellow_phases": {0: 1, 2: 3, 4: 5}
    },
    "Gorko_280015414": {
        "id": "280015414",
        "in_edges": ["49830666#6", "620932850#2", None, "-829563410#1"],
        "out_edges": [None, "-620932850#2", "277523407#1", "829563410#1"],
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
    # Если edge есть - берем данные, если None - пишем 0
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

        # Выделяем каждому перекрестку личные переменные состояния
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

            # 1. Если сейчас горит ЖЕЛТЫЙ
            if current_phase in yellow_dict.values():
                if t - st["last_switch_time"] >= YELLOW_DURATION:
                    # ПРЕДОХРАНИТЕЛЬ: Если SUMO сам включил желтый, а ИИ не был готов
                    if st["target_phase"] is None:
                        st["target_phase"] = green_phases[0]  # Берем первую зеленую фазу для спасения

                    traci.trafficlight.setPhase(tls_id, st["target_phase"])
                    st["last_switch_time"] = t

            # 2. Если сейчас горит ЗЕЛЕНЫЙ
            elif t - st["last_switch_time"] >= MIN_GREEN_TIME and current_phase in green_phases:

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
                desired_phase = green_phases[action_index]

                if desired_phase != current_phase:
                    st["target_phase"] = desired_phase
                    yellow_phase = yellow_dict[current_phase]
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    st["last_switch_time"] = t
                    print(
                        f"[{datetime.timedelta(seconds=t)}] ИИ ({name}) переключает на фазу {desired_phase}. Epsilon: {agent.epsilon:.3f}")
                else:
                    st["last_switch_time"] = t
                    st["target_phase"] = current_phase  # Запоминаем, что мы хотели остаться здесь
                    # ЯВНЫЙ ПРИКАЗ SUMO: "Оставь эту фазу, я перехватываю управление!"
                    traci.trafficlight.setPhase(tls_id, current_phase)
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

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle("Отчет об обучении: Мульти-агентная система", fontsize=16)


        pivot_q = df_traf.pivot_table(index="time", columns="intersection", values="queue", aggfunc='sum')
        pivot_q.plot(ax=axes[0], alpha=0.8, linewidth=2)
        axes[0].set_title("1. Суммарные очереди на въездах в перекрестки")
        axes[0].set_ylabel("Кол-во машин")
        axes[0].grid(True)

        axes[1].set_title("2. Прогресс обучения: Награда (Сглаженный тренд MA 20)")
        for agent_name in df_rl['agent'].unique():
            agent_data = df_rl[df_rl['agent'] == agent_name].copy()
            agent_data['reward_smoothed'] = agent_data['reward'].rolling(window=20, min_periods=1).mean()
            axes[1].plot(agent_data['step'], agent_data['reward_smoothed'], linewidth=2, label=f'{agent_name}')

        axes[1].set_ylabel("Награда (Ближе к 0 = Лучше)")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].set_title("3. Исследование среды (Уровень случайности)")
        for agent_name in df_rl['agent'].unique():
            agent_data = df_rl[df_rl['agent'] == agent_name]
            axes[2].plot(agent_data['step'], agent_data['epsilon'], linewidth=2, label=agent_name)

        axes[2].set_ylabel("Эпсилон")
        axes[2].set_xlabel("Шаги принятия решений")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"Ошибка при построении графиков: {e}")


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