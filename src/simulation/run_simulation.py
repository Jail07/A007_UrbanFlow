import traci
import csv
import pandas as pd
import matplotlib.pyplot as plt

from src.constants.config import SUMO_CFG, LOG_PATH, TLS_ID, SUMO_BINARY

EDGES = {
    "W": ["-622102031#6", "622102031#6"],
    "N": ["-51095930#1", "51095930#1"],
    "E": ["-620932850#1", "620932850#1"],
    "S": ["-580760138#5", "580760138#5"],
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


def get_buses_on_edge(edge_id):
    vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
    buses = [v for v in vehicles if traci.vehicle.getVehicleClass(v) == "bus"]
    for v in vehicles: print(f"ID: {v}, Class: {traci.vehicle.getVehicleClass(v)}")
    bus_routes = [f"{b}:[{len(traci.vehicle.getRoute(b))} edges]" for b in buses]
    return len(buses), "; ".join(bus_routes)


def run_simulation():
    traci.start([SUMO_BINARY, "-c", str(SUMO_CFG)])

    with open(str(LOG_PATH), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "dir", "queue", "vehicles", "mean_speed",
            "lane_count", "lane_configs", "bus_count", "bus_routes"
        ])

        print("Симуляция UrbanFlow запущена...")

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            t = traci.simulation.getTime()

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

                writer.writerow([
                    t, direction, q_total, v_total, avg_speed,
                    total_lanes, " / ".join(all_lane_configs),
                    b_total, " | ".join(all_bus_routes)
                ])

    traci.close()
    print(f"Данные сохранены в {LOG_PATH}")


def analyze_results():
    df = pd.read_csv(str(LOG_PATH))
    print("\nструктура дорог по направлениям")
    infra = df.groupby("dir")[["lane_count", "lane_configs"]].first()
    print(infra)

    print("\nстатистика по направлениям")
    stats = df.groupby("dir").agg({
        "queue": "mean",
        "bus_count": "sum",
        "mean_speed": "mean"
    })
    print(stats)

    pivot_q = df.pivot(index="time", columns="dir", values="queue")
    pivot_q.plot(figsize=(10, 5))
    plt.title("Динамика очередей")
    plt.ylabel("Количество стоящих машин")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        run_simulation()
        analyze_results()
    except KeyboardInterrupt:
        print("Прервано")
        traci.close()
    except traci.exceptions.FatalTraCIError:
        traci.close()