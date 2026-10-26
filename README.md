# Thermal camera person detection

Run `powershell -ExecutionPolicy Bypass -File .\Start-PeopleDetection.ps1` to open live GPU detection. The app selects the USB device named `ThermalCamera`, captures at 640 x 512 / 50 fps, adjusts contrast, and detects people locally on the NVIDIA GPU. Processing uses FP16, a 768-pixel model input, and an 8 GiB GPU memory budget. The window shows the actual detection speed and GPU mode. CUDA must be available; GPU mode reports an error if it cannot start.

The default model is **YOLO11l**, a larger detector with about 25 million parameters. Its official weights are trained on COCO visible-light images, so this is an experimental application to thermal imagery. It is filtered to the person class. Larger model size alone does not establish better accuracy on this camera. Check performance with people at the intended distances, including empty scenes, partial occlusion, and different thermal contrast.

Two thermal-trained alternatives are installed:

| Model file | Description |
| --- | --- |
| `thermal-yolo11n.pt` | Newer thermal person detector, published by banghyunmin, trained on the Roboflow People Detection Thermal dataset. |
| `thermal-person.pt` | Original YOLOv8 nano thermal human detector, published by pitangent-ds. |

Choose one using `powershell -ExecutionPolicy Bypass -File .\Start-PeopleDetection.ps1 -Model thermal-yolo11n.pt`. Model provenance and source links are in `models/SOURCE.md`.

Green boxes show detected people and confidence. The count is the number of boxes in the current frame, not a count of unique visitors. Controls:

- **F**: fullscreen.
- **Q / Esc**: close.
- **+**: increase sensitivity; **-**: decrease sensitivity.
- **I**: invert thermal polarity.
- **R**: rotate the view.

The initial view rotates 90 degrees counterclockwise to match the camera's initially observed orientation. Aim the camera upright with full bodies visible when checking person detection. Reflections can create extra detections. The launcher also supports `-Confidence 0.6`, `-ImageSize 960`, and `-VramGB 8`.

The application does not record video. Current detection status, GPU tensor memory measurements, and diagnostic logs are in `.tools/`. PyTorch GPU memory values do not include the CUDA context or other driver allocations. The allocator is limited to 7 GiB at the default budget, leaving 1 GiB for runtime overhead; this is not an operating-system-wide cap. The original plain viewer is available through `Start-ThermalCamera.ps1` after detection closes. Only one viewer can use the camera at a time.

To recreate the GPU environment, run `powershell -ExecutionPolicy Bypass -File .\Setup-PeopleDetection.ps1`. It downloads uv and FFmpeg when missing, downloads and verifies the model weights, and installs Python 3.12, PyTorch 2.9.1 / torchvision 0.24.1 with CUDA 12.8, and `requirements.txt`. The initial CUDA download is about 2.7 GiB. The CPU environment is preserved separately in `.venv`; the GPU launcher uses `.venv-gpu`. Exact installed GPU versions are recorded in `requirements-gpu-lock.txt`.

Validated on the RTX 5090 Laptop GPU with YOLO11l, FP16, and 768-pixel input: approximately 27-30 FPS in the live viewer and 470 MiB total GPU memory reported by NVIDIA after the benchmark process exited. PyTorch reserved 180 MiB, including a peak of about 137 MiB actively allocated. The reference-image inference benchmark reached about 40 FPS; that excludes USB capture and display. These are performance checks, not measurements of detection accuracy. Memory and speed can change with input size, rotation, and other applications.

For a CPU fallback, run `.\.venv\Scripts\python.exe thermal_people.py --device cpu --model models\thermal-person.pt --imgsz 640`. Ultralytics and its official weights use AGPL-3.0. Refer to each thermal model's source for its stated terms.
