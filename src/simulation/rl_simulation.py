import os

import traci
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import datetime

# --- Импорты для ИИ (PyTorch) ---
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
SUMO_CFG = "data/routes/scenarios_04/sumo.sumocfg"
LOG_PATH = LOGS_DIR / "urbanflow_detailed_log.csv"
RL_LOG_PATH = LOGS_DIR / "rl_training_log.csv"  # НОВЫЙ ЛОГ ДЛЯ ИИ

SUMO_BINARY = "sumo"
# TLS_ID = "244500423"
#
# EDGES = {
#     "W": ["-622102031#6", "622102031#6"],
#     "N": ["-51095930#1", "51095930#1"],
#     "E": ["-620932850#1", "620932850#1"],
#     "S": ["-580760138#5", "580760138#5"]
# }
#
# IN_EDGES = {"W": "622102031#6", "N": "-51095930#1", "E": "-620932850#1", "S": "580760138#5"}
# OUT_EDGES = {"W": "-622102031#6", "N": "51095930#1", "E": "620932850#1", "S": "-580760138#5"}

INTERSECTIONS = {
    "Vefa_244500423": {
        "id": "244500423",
        "in_edges": ["-51095930#1", "622102031#6", "-620932850#1", "580760138#5"],
        "out_edges": ["51095930#1", "-622102031#6", "620932850#1", "-580760138#5"],
        "phases": {"NS_GREEN": 0, "NS_YELLOW": 1, "EW_GREEN": 2, "EW_YELLOW": 3}
    },
    "Kulatov_244500424": {
        "id": "244500424",
        "in_edges": ["-186475564#4", "25684557#4", "186475564#1", "477271462#4"],
        "out_edges": ["186475564#4", "-25684557#4", "-186475564#1", "-477271462#4"],
        "phases": {"NS_left_GREEN": 0, "NS_left_YELLOW": 1, "EW_GREEN": 2, "EW_YELLOW": 3}
    },
    "Muka_280015410": {
        "id": "280015410",
        "in_edges": ["1075767203#0", "-477271462#1", None, "192987238#2"],
        "out_edges": ["-1075767203#0", "477271462#1", "49830666#1", None],
        "phases": {"NS_GREEN": 0, "NS_YELLOW": 1, "EW_GREEN": 2, "EW_YELLOW": 3}
    },
    "Gorko_280015414": {
        "id": "280015414",
        "in_edges": ["49830666#6", "620932850#2", None, "-829563410#1"],
        "out_edges": [None, "-620932850#2", "277523407#1", "829563410#1"],
        "phases": {"EW_GREEN": 0, "EW_YELLOW": 1, "NS_GREEN": 2, "NS_YELLOW": 3}
    }
}


# PHASE_NS_GREEN = 0
# PHASE_NS_YELLOW = 1
# PHASE_EW_GREEN = 2
# PHASE_EW_YELLOW = 3

YELLOW_DURATION = 4
MIN_GREEN_TIME = 15

MAX_SIMULATION_STEPS = 86400


# ==========================================
# 2. АРХИТЕКТУРА ИИ
# ==========================================
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
    def __init__(self, state_size=13, action_size=2):
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

        self.model_path = LOGS_DIR / "hybrid_brain.pth"

    def save_model(self):
        """Сохраняет веса нейросети и текущий Эпсилон"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }, self.model_path)
        print(f"💾 Мозг агента сохранен! (Epsilon: {self.epsilon:.2f})")

    def load_model(self):
        """Загружает память из прошлого запуска, если она есть"""
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
            return None  # Возвращаем None, если еще мало опыта

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

        return total_loss / batch_size  # Возвращаем среднюю ошибку для графиков


# ==========================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_hybrid_state():
    state = []
    for edge in IN_EDGES.values(): state.append(traci.edge.getLastStepHaltingNumber(edge))
    for edge in OUT_EDGES.values(): state.append(traci.edge.getLastStepHaltingNumber(edge))
    for edge in IN_EDGES.values(): state.append(traci.edge.getWaitingTime(edge))
    state.append(traci.trafficlight.getPhase(TLS_ID))
    return np.array(state)


def get_hybrid_reward():
    alpha = 1.0
    beta = 0.05
    in_queue = sum([traci.edge.getLastStepHaltingNumber(e) for e in IN_EDGES.values()])
    out_queue = sum([traci.edge.getLastStepHaltingNumber(e) for e in OUT_EDGES.values()])
    pressure = abs(in_queue - out_queue)
    total_wait_time = sum([traci.edge.getWaitingTime(e) for e in IN_EDGES.values()])
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


# ==========================================
# 4. ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ
# ==========================================
def run_simulation():
    traci.start([SUMO_BINARY, "-c", str(SUMO_CFG)])

    agent = HybridPressureAgent(state_size=13, action_size=2)
    agent.load_model()
    last_switch_time = 0
    target_phase = None
    current_state = None
    last_action = None

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Открываем два лог-файла
    f_traf = open(str(LOG_PATH), "w", newline="")
    f_rl = open(str(RL_LOG_PATH), "w", newline="")

    writer_traf = csv.writer(f_traf)
    writer_rl = csv.writer(f_rl)

    writer_traf.writerow(
        ["time", "dir", "queue", "vehicles", "mean_speed", "lane_count", "lane_configs", "bus_count", "bus_routes"])
    writer_rl.writerow(["step", "reward", "loss", "epsilon"])  # Заголовки для ИИ

    print("Симуляция Hybrid RL Agent запущена...")
    step_counter = 0
    t=0

    while traci.simulation.getMinExpectedNumber() > 0 and t < MAX_SIMULATION_STEPS:
        traci.simulationStep()
        t = int(traci.simulation.getTime())
        current_phase = traci.trafficlight.getPhase(TLS_ID)

        if current_phase in [PHASE_NS_YELLOW, PHASE_EW_YELLOW]:
            if t - last_switch_time >= YELLOW_DURATION:
                if target_phase is None:
                    target_phase = PHASE_EW_GREEN if current_phase == PHASE_NS_YELLOW else PHASE_NS_GREEN
                traci.trafficlight.setPhase(TLS_ID, target_phase)
                last_switch_time = t

        elif t - last_switch_time >= MIN_GREEN_TIME and current_phase in [PHASE_NS_GREEN, PHASE_EW_GREEN]:

            if last_action is not None:
                reward = get_hybrid_reward()
                next_state = get_hybrid_state()
                agent.remember(current_state, last_action, reward, next_state)
                loss = agent.replay(batch_size=32)

                # Логируем прогресс ИИ (каждый раз, когда он принимает решение)
                step_counter += 1
                loss_val = loss if loss is not None else 0
                writer_rl.writerow([step_counter, reward, loss_val, agent.epsilon])

            current_state = get_hybrid_state()
            action = agent.act(current_state)
            last_action = action

            desired_phase = PHASE_NS_GREEN if action == 0 else PHASE_EW_GREEN

            if desired_phase != current_phase:
                target_phase = desired_phase
                yellow_phase = PHASE_NS_YELLOW if current_phase == PHASE_NS_GREEN else PHASE_EW_YELLOW
                traci.trafficlight.setPhase(TLS_ID, yellow_phase)
                last_switch_time = t
                print(f"[{datetime.timedelta(seconds=t)}c] ИИ переключает на фазу {desired_phase}. Epsilon: {agent.epsilon:.3f}")

        # Сбор метрик трафика (каждые 10 секунд, чтобы не перегружать лог)
        if t % 10 == 0:
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

                avg_speed = round((s_sum / len(edges)) * 3.6, 2) if v_total > 0 else 0
                writer_traf.writerow(
                    [t, direction, q_total, v_total, avg_speed, total_lanes, " / ".join(all_lane_configs), 0, ""])

    f_traf.close()
    f_rl.close()
    agent.save_model()
    traci.close()
    print("Симуляция завершена. Логи сохранены.")


# ==========================================
# 5. АНАЛИЗ РЕЗУЛЬТАТОВ (Дашборд обучения)
# ==========================================
def analyze_results():
    try:
        df_traf = pd.read_csv(str(LOG_PATH))
        df_rl = pd.read_csv(str(RL_LOG_PATH))

        # Создаем фигуру с 3 подграфиками
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle("Отчет об обучении: Hybrid RL Agent", fontsize=16)

        # График 1: Динамика очередей (Трафик)
        pivot_q = df_traf.pivot(index="time", columns="dir", values="queue")
        pivot_q.plot(ax=axes[0], alpha=0.7)
        axes[0].set_title("1. Динамика очередей на перекрестке")
        axes[0].set_ylabel("Кол-во машин")
        axes[0].grid(True)

        # График 2: Полученная Награда (Сглаженная)
        # Сглаживаем награду (скользящее среднее), чтобы видеть тренд, а не скачки
        df_rl['reward_smoothed'] = df_rl['reward'].rolling(window=20, min_periods=1).mean()
        axes[1].plot(df_rl['step'], df_rl['reward'], alpha=0.3, color='red', label='Сырая награда')
        axes[1].plot(df_rl['step'], df_rl['reward_smoothed'], color='darkred', linewidth=2, label='Тренд (MA 20)')
        axes[1].set_title("2. Прогресс обучения: Награда (Reward)")
        axes[1].set_ylabel("Награда (Ближе к 0 = Лучше)")
        axes[1].legend()
        axes[1].grid(True)

        # График 3: Падение Эпсилона (Исследование -> Использование)
        axes[2].plot(df_rl['step'], df_rl['epsilon'], color='green', linewidth=2)
        axes[2].set_title("3. Исследование среды (Epsilon Decay)")
        axes[2].set_ylabel("Уровень случайности действий")
        axes[2].set_xlabel("Шаги принятия решений")
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
        analyze_results()  # Показываем графики, даже если прервали по Ctrl+C
    except traci.exceptions.FatalTraCIError as e:
        print(f"TraCI Error: {e}")
        traci.close()