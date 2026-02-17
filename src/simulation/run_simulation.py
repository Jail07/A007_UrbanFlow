import traci
import csv

SUMO_CFG = "D:/Users/Admin/PycharmProjects/UrbanFlow/sumo/configs/sumo.sumocfg"
TLS_ID = "244500423"

EDGES = {
    "W": ["-622102031#6", "622102031#6"],
    "N": ["-51095930#1", "51095930#1"],
    "E": ["-620932850#1", "620932850#1"],
    "S": ["-580760138#5", "580760138#5"],
}

traci.start(["sumo", "-c", SUMO_CFG])

with open("D:/Users/Admin/PycharmProjects/UrbanFlow/data/logs/intersection_244500423.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time", "dir",
        "queue", "vehicles",
        "mean_speed"
    ])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t = traci.simulation.getTime()

        for d, edges in EDGES.items():

            queue = sum(
                traci.edge.getLastStepHaltingNumber(e)
                for e in edges
            )

            vehs = sum(
                traci.edge.getLastStepVehicleNumber(e)
                for e in edges
            )

            speed = sum(
                traci.edge.getLastStepMeanSpeed(e)
                for e in edges
            ) / len(edges)


            writer.writerow([t, d, queue, vehs, speed])

total_queue = sum(
        traci.edge.getLastStepHaltingNumber(e)
        for edges in EDGES.values()
        for e in edges
    )
print(total_queue)
traci.close()


