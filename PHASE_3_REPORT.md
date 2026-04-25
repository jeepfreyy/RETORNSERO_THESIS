# Phase 3 Progress Report: Dual-Camera Stress Testing & Optimization

## 1. Overview
We have officially initiated **Phase 3** of the Retornsero project. The core objective of this phase is to validate the system's **Dual-Camera Architecture** under realistic, heavy-load conditions. This is a critical component of the thesis, as it differentiates the system from simple single-stream classical CV setups and proves its viability for multi-angle nighttime surveillance.

## 2. Changes Implemented
- **Video Source Update (`app.py`)**: 
  - Modified the initialization parameters for `cam1_stream` and `cam2_stream`.
  - Replaced the single placeholder `video1.mp4` with two distinct high-resolution video files:
    - **CAM-01**: `videos/vid1-angle1.MOV` (~500MB)
    - **CAM-02**: `videos/vid2-angle2.MOV` (~500MB)
- **Concurrent Processing Integration**:
  - By feeding both heavy `.MOV` files into the `SentinelStream` instances, we are now actively stressing the Python Global Interpreter Lock (GIL) and system memory, forcing the backend to handle concurrent frame extraction, background subtraction, and object tracking.

## 3. Technical Walkthrough: Dual-Camera Concurrency

### How it works under the hood
1. **Initialization (`app.py`)**: When the Flask app boots, it spins up two instances of the `SentinelStream` class. 
2. **Background Threads (`vision_engine.py`)**: Each `SentinelStream` immediately spawns its own isolated background thread (`self.thread = threading.Thread(target=self.update, args=())`). This is crucial because it decouples the heavy computer vision processing from the Flask web server, ensuring the UI remains responsive.
3. **Independent Pipelines**: 
   - Each thread opens its own `cv2.VideoCapture` object pointing to the respective `.MOV` file.
   - They independently apply the `mask_layer.png` to crop out irrelevant background elements.
   - They independently run contour detection and the `RobustSentinelTracker` algorithm.
4. **Data Delivery**: As the frontend requests frames (`/cam1_frame` and `/cam2_frame`), it pulls from the `self.frame` buffer of each respective stream.

## 4. Next Steps for Phase 3
To fully realize the goals of this phase, we should consider implementing the following:
1. **Independent Hyperparameter Tuning**: Currently, both cameras might be using the exact same detection thresholds. We need to allow CAM-01 and CAM-02 to have distinct sensitivity parameters (e.g., area threshold, ghost threshold) so they can be optimized for their specific lighting conditions.
2. **Performance Profiling**: Monitor the CPU/RAM usage while both ~500MB videos are running to ensure the system doesn't bottleneck or drop frames significantly.
3. **Dynamic Masking**: Allow CAM-02 to use a different background mask (`mask_layer2.png`) since its angle might cover different restricted zones.
