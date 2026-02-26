import traci

SUMO_CFG = "D:/Users/Admin/PycharmProjects/UrbanFlow/data/routes/sumo.sumocfg"
TLS_ID = "244500423"

EDGES = {
    "N": ["-51095930#1", "51095930#1"],
    "S": ["-580760138#5", "580760138#5"],
    "E": ["-620932850#1", "620932850#1"],
    "W": ["-622102031#6", "622102031#6"],
}

PHASE_NS = 0
PHASE_EW = 2

MIN_GREEN = 15


def queue(edges):
    return sum(traci.edge.getLastStepHaltingNumber(e) for e in edges)

traci.start(["sumo-gui", "-c", SUMO_CFG])

current_phase = PHASE_NS
traci.trafficlight.setPhase(TLS_ID, current_phase)

last_switch = 0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    t = traci.simulation.getTime()

    if t - last_switch < MIN_GREEN:
        continue

    q_ns = queue(EDGES["N"] + EDGES["S"])
    q_ew = queue(EDGES["E"] + EDGES["W"])

    new_phase = PHASE_NS if q_ns >= q_ew else PHASE_EW

    if new_phase != current_phase:
        traci.trafficlight.setPhase(TLS_ID, new_phase)
        current_phase = new_phase
        last_switch = t

traci.close()