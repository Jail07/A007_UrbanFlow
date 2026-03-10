from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
ROUTES_DIR = DATA_DIR / "routes"
NETWORKS_DIR = DATA_DIR / "network"
OSM_DIR = DATA_DIR / "osm"
VIDEO_DIR = DATA_DIR / "video"
YOLOV3_DIR = DATA_DIR / "yolov3"

SUMO_CFG = ROUTES_DIR / "sumo.sumocfg"
LOG_PATH = LOGS_DIR / "urbanflow_detailed_log.csv"
DF_PATH = LOGS_DIR / "intersection_244500423.csv"
VIDEO_PATH = VIDEO_DIR / "video_0.mp4"
VIDEO_PATH_1 = VIDEO_DIR / "video_1.mp4"


YOLOV3_CFG_PATH = YOLOV3_DIR / "yolov3.cfg"
YOLOV3_WEIGHTS_PATH = YOLOV3_DIR / "yolov3.weights"

SUMO_BINARY = "sumo-gui"
# SUMO_BINARY = "sumo"
TLS_ID = "244500423"