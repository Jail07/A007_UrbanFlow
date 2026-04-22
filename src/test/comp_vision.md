# Computer Vision Comparison: UrbanFlow

This section contains the results of a comparative analysis of two versions of computer vision systems for monitoring urban traffic.

## 📝 Project Description

The **UrbanFlow** project is aimed at analyzing traffic flows in real time. The `src/test` folder contains two iterations of object detection and tracking algorithms.

*   **Version 1 (`computer_vision_test.py`)**: Basic detection based on YOLOv3.
*   **Version 2 (`computer_vision_test_v2.py`)**: Advanced system based on YOLOv8 using the DeepSort tracker.

## 🚀 How to Run

### Prerequisites
Make sure the required dependencies are installed:
```bash
pip install ultralytics deep_sort_realtime opencv-python numpy
```

### Running the Scripts
From the project root directory:
```bash
# Run v1 (YOLOv3)
python -m src.test.computer_vision_test

# Run v2 (YOLOv8 + DeepSort)
python -m src.test.computer_vision_test_v2
```

## 📊 Results Table

| Feature | Version 1 (YOLOv3) | Version 2 (YOLOv8) |
| :--- | :--- | :--- |
| **Architecture** | YOLOv3 (OpenCV DNN) | YOLOv8n (Ultralytics) |
| **Tracking** | None | DeepSort Tracker |
| **Confidence Threshold** | 0.5 | 0.5 |
| **Class Filtering** | All COCO classes | Only transport (2, 3, 5, 7) |
| **Performance** | Medium (Legacy CPU) | High (Real-time) |
| **ID Persistence** | No (new IDs every frame) | Yes (Track ID persistence) |
| **Bounding Box Format** | Center-based (computed) | XYXY (native) |

## 💾 CSV Example

The results are also exported in CSV format. You can find them in `src/test/results.csv`.

**Example structure:**
```csv
Metric, Version_1 (YOLOv3), Version_2 (YOLOv8)
Architecture, YOLOv3 (OpenCV DNN), YOLOv8n (Ultralytics)
Tracking, Manual Persistence (None), DeepSort Tracker
...
```

## 📈 Conclusion (Comparison)

**Version 2 is significantly more advanced for the following reasons:**
1.  **Tracking availability**: DeepSort allows identifying a specific vehicle throughout the entire video, rather than simply detecting it as "some object" in each frame.
2.  **Speed and accuracy**: YOLOv8n (nano) is significantly faster than the outdated YOLOv3 and provides better results on small object detection.
3.  **Specialization**: v2 is configured to filter only vehicles, which is critical for UrbanFlow tasks.

> [!TIP]
> For production use, it is recommended to use Version 2, as it provides a foundation for counting unique vehicles and analyzing trajectories.
