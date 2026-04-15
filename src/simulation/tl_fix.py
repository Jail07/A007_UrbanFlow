import traci
import csv
import pandas as pd
from pathlib import Path
import datetime

# --- НАСТРОЙКИ ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
SUMO_CFG = "data/routes/scenarios_04/sumo.sumocfg"

FIXED_LOG_PATH = LOGS_DIR / "log_fixed.csv"
SUMO_BINARY = "sumo-gui"
MAX_SIMULATION_STEPS = 86400

INTERSECTIONS = {
    "Vefa_244500423": {
        "id": "244500423",
        "in_edges": ["-51095930#1", "622102031#6", "-620932850#1", "580760138#5"]
    },
    "Kulatov_244500424": {
        "id": "244500424",
        "in_edges": ["-186475564#4", "25684557#4", "186475564#1", "477271462#4"]
    }
}


def get_lane_info(edge_id):
    num_lanes = traci.edge.getLaneNumber(edge_id)
    lane_data = []
    for i in range(num_lanes):
        lane_id = f"{edge_id}_{i}"
        links = traci.lane.getLinks(lane_id)
        directions = "".join(sorted(set(link[6] for link in links)))
        lane_data.append(f"L{i}({directions})")
    return num_lanes, "|".join(lane_data)


def run_fixed_simulation():
    # Запускаем SUMO
    traci.start([SUMO_BINARY, "-c", str(SUMO_CFG)])

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    f_traf = open(str(FIXED_LOG_PATH), "w", newline="")
    writer_traf = csv.writer(f_traf)

    writer_traf.writerow(
        ["time", "intersection", "queue", "vehicles", "mean_speed", "lane_count", "lane_configs", "bus_count",
         "bus_routes"]
    )

    print("Симуляция фиксированных светофоров (Baseline) запущена...")
    t = 0

    # Главный цикл
    while traci.simulation.getMinExpectedNumber() > 0 and t < MAX_SIMULATION_STEPS:
        # Продвигаем время. SUMO САМ переключает светофоры по своим внутренним таймерам
        traci.simulationStep()
        t = int(traci.simulation.getTime())

        # Каждые 10 секунд собираем статистику
        if t % 10 == 0:
            for name, data in INTERSECTIONS.items():
                in_edges = data["in_edges"]
                q_total, v_total, s_sum, total_lanes = 0, 0, 0, 0
                all_lane_configs = []

                for e in in_edges:
                    q_total += traci.edge.getLastStepHaltingNumber(e)
                    v_total += traci.edge.getLastStepVehicleNumber(e)
                    s_sum += traci.edge.getLastStepMeanSpeed(e)

                    n_lanes, l_cfg = get_lane_info(e)
                    total_lanes += n_lanes
                    all_lane_configs.append(l_cfg)

                avg_speed = round((s_sum / len(in_edges)) * 3.6, 2) if v_total > 0 else 0

                writer_traf.writerow(
                    [t, name, q_total, v_total, avg_speed, total_lanes, " / ".join(all_lane_configs), 0, ""]
                )

            if t % 1000 == 0:
                print(f"Прошло времени: {datetime.timedelta(seconds=t)}")

    f_traf.close()
    traci.close()
    print(f"Симуляция завершена. Базовые логи сохранены в {FIXED_LOG_PATH}")


if __name__ == "__main__":
    try:
        run_fixed_simulation()
    except KeyboardInterrupt:
        print("Прервано пользователем")
        traci.close()