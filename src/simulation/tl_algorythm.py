import traci
import csv
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
ROUTES_DIR = DATA_DIR / "routes"
NETWORKS_DIR = DATA_DIR / "network"
OSM_DIR = DATA_DIR / "osm"
SUMO_CFG = "data/routes/scenarios_04/sumo.sumocfg"
LOG_PATH = LOGS_DIR / "algorythm_logs.csv"
RL_LOG_PATH = LOGS_DIR / "algorythm_training_log.csv"


SUMO_BINARY = "sumo-gui"
# SUMO_BINARY = "sumo"
TLS_ID = "244500423"
EDGES = {
    "W": ["-622102031#6", "622102031#6"],
    "N": ["-51095930#1", "51095930#1"],
    "E": ["-620932850#1", "620932850#1"],
    "S": ["-580760138#5", "580760138#5"]}


IN_EDGES = {
    "W": "622102031#6",
    "N": "-51095930#1",
    "E": "-620932850#1",
    "S": "580760138#5",
}

OUT_EDGES = {
    "W": "-622102031#6",
    "N": "51095930#1",
    "E": "620932850#1",
    "S": "-580760138#5",
}

PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2
PHASE_NS_YELLOW = 1
PHASE_EW_YELLOW = 3
YELLOW_DURATION = 4


def get_edge_queue(edge_id):
    """Получает количество стоящих машин на ребре."""
    return traci.edge.getLastStepHaltingNumber(edge_id)


def calculate_phase_pressure():

    q_in_n = get_edge_queue(IN_EDGES["N"])
    q_in_s = get_edge_queue(IN_EDGES["S"])
    q_in_e = get_edge_queue(IN_EDGES["E"])
    q_in_w = get_edge_queue(IN_EDGES["W"])

    q_out_n = get_edge_queue(OUT_EDGES["N"])
    q_out_s = get_edge_queue(OUT_EDGES["S"])
    q_out_e = get_edge_queue(OUT_EDGES["E"])
    q_out_w = get_edge_queue(OUT_EDGES["W"])

    pressure_ns = (q_in_n + q_in_s) - (q_out_n + q_out_s)
    pressure_ew = (q_in_e + q_in_w) - (q_out_e + q_out_w)

    return pressure_ns, pressure_ew

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
    # for v in vehicles: print(f"ID: {v}, Class: {traci.vehicle.getVehicleClass(v)}")
    bus_routes = [f"{b}:[{len(traci.vehicle.getRoute(b))} edges]" for b in buses]
    return len(buses), "; ".join(bus_routes)


def run_simulation():
    traci.start([SUMO_BINARY, "-c", str(SUMO_CFG)])

    last_switch_time = 0
    min_green_time = 15

    f_traf = open(str(LOG_PATH), "w", newline="")
    f_rl = open(str(RL_LOG_PATH), "w", newline="")

    writer_traf = csv.writer(f_traf)
    writer_rl = csv.writer(f_rl)

    writer_traf.writerow(
        ["time", "dir", "queue", "vehicles", "mean_speed", "lane_count", "lane_configs", "bus_count", "bus_routes"])
    writer_rl.writerow(["step", "reward", "loss", "epsilon"])  # Заголовки для ИИ

    print("Симуляция UrbanFlow запущена...")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t = traci.simulation.getTime()

        current_phase = traci.trafficlight.getPhase(TLS_ID)

        if current_phase == PHASE_NS_YELLOW:
            if t - last_switch_time >= YELLOW_DURATION:
                traci.trafficlight.setPhase(TLS_ID, PHASE_EW_GREEN)
                last_switch_time = t

        elif current_phase == PHASE_EW_YELLOW:
            if t - last_switch_time >= YELLOW_DURATION:
                traci.trafficlight.setPhase(TLS_ID, PHASE_NS_GREEN)
                last_switch_time = t

        elif t - last_switch_time > min_green_time and current_phase in [PHASE_NS_GREEN, PHASE_EW_GREEN]:

            pressure_ns, pressure_ew = calculate_phase_pressure()

            if int(t) % 10 == 0:
                print(f"Время: {int(t)}с | Давление NS: {pressure_ns} | Давление EW: {pressure_ew}")

            # Переключаем, если чужое давление больше
            if pressure_ew > pressure_ns and current_phase == PHASE_NS_GREEN:
                traci.trafficlight.setPhase(TLS_ID, PHASE_NS_YELLOW)  # Сначала желтый!
                last_switch_time = t
                print(f"--- Переключение на EW (Желтый) ---")

            elif pressure_ns >= pressure_ew and current_phase == PHASE_EW_GREEN:
                traci.trafficlight.setPhase(TLS_ID, PHASE_EW_YELLOW)  # Сначала желтый!
                last_switch_time = t
                print(f"--- Переключение на NS (Желтый) ---")

        for direction, edges in EDGES.items():
            q_total = 0
            v_total = 0
            s_sum = 0
            b_total = 0

            all_lane_configs = []
            all_bus_routes = []
            total_lanes = 0

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

    traci.close()
    print(f"Данные сохранены в {LOG_PATH}")


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


        df_rl['reward_smoothed'] = df_rl['reward'].rolling(window=20, min_periods=1).mean()
        axes[1].plot(df_rl['step'], df_rl['reward'], alpha=0.3, color='red', label='Сырая награда')
        axes[1].plot(df_rl['step'], df_rl['reward_smoothed'], color='darkred', linewidth=2, label='Тренд (MA 20)')
        axes[1].set_title("2. Прогресс обучения: Награда (Reward)")
        axes[1].set_ylabel("Награда (Ближе к 0 = Лучше)")
        axes[1].legend()
        axes[1].grid(True)


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
        print("Прервано")
        traci.close()
    except traci.exceptions.FatalTraCIError:
        traci.close()