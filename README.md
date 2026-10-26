ï»¿# Thermal camera person detection

Run `powershell -ExecutionPolicy Bypass -File .\Setup-PeopleDetection.ps1` once, then `powershell -ExecutionPolicy Bypass -File .\Start-PeopleDetection.ps1` for live detection on the NVIDIA GPU. Setup downloads the camera tools, verifies the thermal weights, prepares the pinned YOLOv5 runtime, and installs CUDA-enabled PyTorch. The initial CUDA package download is about 2.7 GiB.

The default is **thermal-trained DETR ResNet-50 with 41,502,152 parameters**, fine-tuned on the Roboflow Thermal Dogs and People dataset. A separate **thermal YOLOv8n person model with 3,011,043 parameters** checks its boxes. A box is shown only when both models overlap by at least 0.25 IoU and the box appears in three consecutive frames. Default primary confidence is 0.65 and verifier confidence is 0.4. This replaces the legacy YOLOv5x default after small false boxes were reported on camera backgrounds. Checkpoints must match their recorded source revisions and SHA-256 hashes; general-photo checkpoints are rejected.

The publisher's notebook explicitly maps person to class 2. Its thermal fine-tuning set is small (142 training images), and this checkpoint has not been fine-tuned on this particular camera. See `models/SOURCE.md` for provenance and limitations.

The USB camera supplies **640 x 512 at 50 fps**. At the default settings, each source frame produces:

1. One full-frame view, padded to 640 x 640 without shrinking any source pixels.
2. Four 384 x 384 crops taken directly from the source image, with at least 128 pixels of overlap. Each crop is enlarged to the model's 640 x 640 input.
3. Each detector processes all five views as a GPU batch. Boxes are mapped back to full-frame coordinates, deduplicated, checked between models, then confirmed across frames. The two models execute sequentially on the same GPU.

The full-frame view helps with people who cross crop boundaries. Crops let smaller people occupy more of the model input. Upsampling does not create extra sensor detail. Every native pixel remains covered, and inputs that would shrink source pixels are rejected. DETR normalization runs on the GPU and matches the official processor's normalization, mask, and padding; box decoding uses the official processor. Requiring agreement reduces unsupported boxes but can miss people when either model fails. Overlapping or partially visible people can still cause missed detections or duplicate boxes.

Green boxes show detected people and confidence. The count is the number of retained boxes in the current frame, not a count of unique visitors. Controls:

- **F**: fullscreen.
- **Q / Esc**: close.
- **+**: increase sensitivity; **-**: decrease sensitivity.
- **I**: invert thermal polarity.
- **R**: rotate the view.

The initial view rotates 90 degrees counterclockwise. The launcher supports `-Confidence 0.65`, `-ConfirmFrames 3`, `-TileSize 384`, `-TileOverlap 128`, `-BatchSize 5`, `-ImageSize 640`, and `-VramGB 8`. Use `-NoTiles` for only the full-frame pass. Batch size is automatically reduced if PyTorch runs out of allocator space, while retaining every view. Confirmation introduces about two inference intervals of delay and removes a box immediately when agreement disappears.

The legacy 88.5M-parameter thermal YOLOv5x remains available with `-Model thermal-yolov5x-adam.pt`. Other primary checkpoints are `thermal-yolo11n.pt` and `thermal-person.pt`; use `-VerifyModel` to choose a different independent thermal verifier. The default pairing is DETR plus `thermal-person.pt`.

Measured on the RTX 5090 Laptop GPU with both models in FP16, 640-pixel input and five views each: approximately **8 FPS and 946 MiB total GPU memory (0.92 GiB)**. PyTorch reserved 654 MiB with a peak of 428 MiB actively allocated. These are local performance measurements, not detection-accuracy scores. A saved camera-background check produced no primary person candidates. A separate thermal reference image retained person boxes, but still showed missed/duplicate detections. Camera-specific false-positive rate and recall have not been measured.

The GPU allocator is capped at 7 GiB at the default 8 GiB budget, leaving 1 GiB for CUDA runtime overhead. This is not an operating-system-wide memory cap. Current status and memory measurements are in `.tools/detection-status.json`; performance can vary with image size, batch size, and other applications.

The application does not record video. Camera snapshots used for diagnostics, downloaded binaries, weights, and virtual environments are ignored by Git. The original plain viewer remains available through `Start-ThermalCamera.ps1` after detection closes. Only one viewer can open the camera at a time.

Run `.\.venv-gpu\Scripts\python.exe -m unittest test_thermal_tiling test_thermal_confirmation -v` for 12 checks covering complete source-pixel coverage, batched crop processing, coordinate mapping, duplicate suppression, memory-budget fallback, rejection of general-photo weights, independent model agreement, consecutive-frame confirmation, official DETR box decoding, and processor-equivalent normalization/padding. Actual GPU inference was checked on saved thermal background and person-reference images.

The GPU environment uses Python 3.12 and PyTorch 2.9.1 / torchvision 0.24.1 with CUDA 12.8. Exact installed packages are recorded in `requirements-gpu-lock.txt`. Model and dependency license information is in `models/SOURCE.md`.
