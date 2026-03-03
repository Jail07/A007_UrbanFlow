import traci
import time

SUMO_BINARY = "sumo-gui"
SUMO_CFG = "sumo/sumo.sumocfg"

def get_queue(edge_id):
    return traci.edge.getLastStepHaltingNumber(edge_id)

def main():
    traci.start([SUMO_BINARY, "-c", SUMO_CFG])

    tls_id = "c"
    north_edge = "n2c"
    south_edge = "s2c"

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        q_north = get_queue(north_edge)
        q_south = get_queue(south_edge)

        if q_north > q_south:
            traci.trafficlight.setPhase(tls_id, 0)  # зелёный север -> юг
        else:
            traci.trafficlight.setPhase(tls_id, 2)  # зелёный юг -> север

        # time.sleep(0.2)

    traci.close()

if __name__ == "__main__":
    main()
