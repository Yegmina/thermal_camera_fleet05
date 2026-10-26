# Thermal camera person detection

Run `powershell -ExecutionPolicy Bypass -File .\Setup-PeopleDetection.ps1` once, then `powershell -ExecutionPolicy Bypass -File .\Start-PeopleDetection.ps1` for live detection on the NVIDIA GPU. Setup downloads the camera tools, verifies the thermal weights, prepares the pinned YOLOv5 runtime, and installs CUDA-enabled PyTorch. The initial CUDA package download is about 2.7 GiB.

The default is **thermal-trained YOLOv5x with 88,474,028 parameters**, published by MAli-Farooq. The publisher's saved training configuration specifies the thermal dataset `thermal11.yaml`, 640 x 640 input, and 100 configured epochs. The checkpoint explicitly labels `person` as class 5. Its source revision, archive member, and SHA-256 are recorded in `models/manifest.json` and `models/SOURCE.md`. Every checkpoint must match an approved thermal entry and checksum. General-photo checkpoints are rejected.

The USB camera supplies **640 x 512 at 50 fps**. At the default settings, each source frame produces:

1. One full-frame view, padded to 640 x 640 without shrinking any source pixels.
2. Four 384 x 384 crops taken directly from the source image, with at least 128 pixels of overlap. Each crop is enlarged to the model's 640 x 640 input.
3. A batch of all five views on the GPU. Boxes are mapped back to full-frame coordinates and combined with non-maximum suppression to remove overlapping duplicates.

The full-frame view helps with people who cross crop boundaries. Crops let smaller people occupy more of the model input. Upsampling does not create extra sensor detail. Every native pixel remains covered, and inputs that would shrink source pixels are rejected. Overlapping or partially visible people can still cause missed detections or duplicate boxes.

Green boxes show detected people and confidence. The count is the number of retained boxes in the current frame, not a count of unique visitors. Controls:

- **F**: fullscreen.
- **Q / Esc**: close.
- **+**: increase sensitivity; **-**: decrease sensitivity.
- **I**: invert thermal polarity.
- **R**: rotate the view.

The initial view rotates 90 degrees counterclockwise. The launcher supports `-Confidence 0.6`, `-TileSize 384`, `-TileOverlap 128`, `-BatchSize 5`, `-ImageSize 640`, and `-VramGB 8`. Use `-NoTiles` for only the full-frame pass. Batch size is automatically reduced if PyTorch runs out of allocator space, while retaining every view.

Two smaller thermal-trained alternatives remain available: `-Model thermal-yolo11n.pt` or `-Model thermal-person.pt`. They are filtered to the named person/human class and must pass the same checkpoint checksum checks.

Measured on the RTX 5090 Laptop GPU with FP16, 640-pixel input, the full-frame pass and four crops in a batch of five: approximately **16 FPS and 944 MiB total GPU memory**. PyTorch reserved about 662 MiB, with a peak of about 429 MiB actively allocated. These are local performance measurements, not detection-accuracy scores. Validate people and empty scenes at the intended distances; the model is thermal-trained but has not been fine-tuned on this specific camera.

The GPU allocator is capped at 7 GiB at the default 8 GiB budget, leaving 1 GiB for CUDA runtime overhead. This is not an operating-system-wide memory cap. Current status and memory measurements are in `.tools/detection-status.json`; performance can vary with image size, batch size, and other applications.

The application does not record video. Camera snapshots used for diagnostics, downloaded binaries, weights, and virtual environments are ignored by Git. The original plain viewer remains available through `Start-ThermalCamera.ps1` after detection closes. Only one viewer can open the camera at a time.

Run `.\.venv-gpu\Scripts\python.exe -m unittest test_thermal_tiling -v` for checks covering complete source-pixel coverage, batched crop processing, coordinate mapping, duplicate suppression, memory-budget fallback, and rejection of general-photo weights. Actual GPU inference was also checked against the publisher's thermal example with valid person boxes.

The GPU environment uses Python 3.12 and PyTorch 2.9.1 / torchvision 0.24.1 with CUDA 12.8. Exact installed packages are recorded in `requirements-gpu-lock.txt`. Model and dependency license information is in `models/SOURCE.md`.
