import csv
import os

def export_cv_results():
    # Data derived from analysis of computer_vision_test.py and computer_vision_test_v2.py
    results = [
        ("Architecture", "YOLOv3 (OpenCV DNN)", "YOLOv8n (Ultralytics)"),
        ("Tracking", "Manual Persistence (None)", "DeepSort Tracker"),
        ("Confidence Threshold", "0.5", "0.5"),
        ("Class Filtering", "None (All COCO)", "Vehicles (Car, Motor, Bus, Truck)"),
        ("Library Dependency", "opencv-python, numpy", "ultralytics, deep_sort_realtime"),
        ("Performance Tier", "Legacy / CPU-heavy", "State-of-the-Art / Real-time"),
        ("ID Persistence", "No (New ID every frame)", "Yes (Consistent Track IDs)"),
        ("Bounding Box Format", "Center-based (calculated)", "XYXY (native)"),
    ]

    output_file = "src/test/results.csv"
    
    # Ensure directory exists (though src/test should exist)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Version_1 (YOLOv3)", "Version_2 (YOLOv8)"])
        writer.writerows(results)

    print(f"Results successfully exported to {output_file}")

if __name__ == "__main__":
    export_cv_results()
