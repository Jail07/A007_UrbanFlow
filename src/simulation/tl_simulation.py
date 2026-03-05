import traci
import csv
from pathlib import Path

TLS_ID = "244500423"
SUMO_BINARY = "sumo"
BASE_DIR = Path(__file__).resolve().parents[2]
SUMO_CFG = BASE_DIR / "data/routes/sumo.sumocfg"

LOG_DIR = Path("experiments/exp_with_tl")
LOG_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    (20,10,3),
    (25,10,3),
    (30,10,3),
    (20,15,3),
    (25,15,3),
    (30,15,3),
    (35,15,3),
    (20,20,3),
    (25,20,3),
    (30,20,3),
    (35,20,3),
    (40,20,3),
    (20,25,3),
    (25,25,3),
    (30,25,3),
    (35,25,3),
    (40,25,3),
    (30,30,3),
    (35,30,3),
    (40,30,3)
]

def apply_tls_logic(green_main, green_side, yellow):

    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]

    phases = logic.phases

    new_phases = []

    for i, p in enumerate(phases):

        duration = p.duration

        if i == 0:
            duration = green_main
        elif i == 1:
            duration = yellow
        elif i == 2:
            duration = green_side
        elif i == 3:
            duration = yellow

        new_phases.append(
            traci.trafficlight.Phase(duration=duration, state=p.state)
        )

    logic.phases = new_phases

    traci.trafficlight.setProgramLogic(TLS_ID, logic)


def run_experiment(exp_id, green_main, green_side, yellow):

    log_path = LOG_DIR / f"exp_{exp_id}.csv"

    traci.start([SUMO_BINARY, "-c", str(SUMO_CFG)])

    apply_tls_logic(green_main, green_side, yellow)

    with open(log_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["time","vehicles","queue","speed"])

        while traci.simulation.getMinExpectedNumber() > 0:

            traci.simulationStep()

            t = traci.simulation.getTime()

            queue = 0
            vehicles = 0
            speed = 0

            edges = [
                "-622102031#6",
                "622102031#6",
                "-51095930#1",
                "51095930#1"
            ]

            for e in edges:
                queue += traci.edge.getLastStepHaltingNumber(e)
                vehicles += traci.edge.getLastStepVehicleNumber(e)
                speed += traci.edge.getLastStepMeanSpeed(e)

            writer.writerow([t,vehicles,queue,speed])

    traci.close()


def main():

    for i, params in enumerate(EXPERIMENTS):

        green_main, green_side, yellow = params

        print(f"experiment {i+1}/20  main={green_main} side={green_side}")

        run_experiment(i+1, green_main, green_side, yellow)


if __name__ == "__main__":
    main()
