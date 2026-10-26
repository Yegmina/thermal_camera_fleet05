# Thermal checkpoint sources

## Default: thermal YOLOv5x

- Publisher: [MAli-Farooq/Thermal_Object_Detection](https://huggingface.co/MAli-Farooq/Thermal_Object_Detection).
- Related code and research: [Thermal-YOLO and model optimization](https://github.com/MAli-Farooq/Thermal-YOLO-And-Model-Optimization-Using-TensorFlowLite).
- Source revision: `0df98a0ed9e200be78a313f5f74d347a90a93a3c`.
- Archive: `Heliaus-Thermal-SGD-ADAM.zip`.
- Selected member: `v2/ADAM/ADAM-Xlarge/weights/best.pt`.
- Local filename: `thermal-yolov5x-adam.pt`.
- SHA-256: `2510bdbe3e294317b58c7f57ed315ffd46550d3354c9030652a04a31d29fb618`.
- Checkpoint size: 177,550,789 bytes; 88,474,028 parameters before inference-layer fusion.
- Checkpoint labels: `bicycle`, `bike`, `bus`, `car`, `dog`, `person`, `pole`.
- Publisher's saved `opt.yaml`: `data: ./data/thermal11.yaml`, `img_size: [640, 640]`, `epochs: 100`, `adam: true`. A copy is preserved in `thermal-yolov5x-training.yaml`.
- Thermal-training provenance: the publisher describes the archive as YOLOv5 variants tuned on thermal data. Its supplied reference images show thermal person and vehicle annotations. The exact sensor-specific training distribution is not fully documented here.
- Model repository license stated by publisher: MIT. The research code repository and YOLOv5 runtime are GPL-3.0; the model-card license alone should not be treated as licensing all dependencies.

`download_thermal_model.py` downloads only the selected member from the pinned archive using an HTTP byte range, then verifies decompression length, ZIP CRC32, and SHA-256. `thermal_detector.py` verifies SHA-256 again before loading the legacy PyTorch checkpoint.

The pinned inference runtime is [Ultralytics YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0), downloaded to `.tools/yolov5-7.0` with its upstream GPL-3.0 license. Archive SHA-256: `a40568a7979c4b27195d198987e49610de86b038b044f2521f5515455571f852`. `thermal_runtime.py` defers its optional Seaborn plotting import; this does not change the model's inference operations. The loader adapts old activation/Upsample attributes and rebuilds both detection-grid caches for compatibility with the pinned runtime.

## Smaller thermal alternatives

- `thermal-yolo11n.pt`: `best.pt` from [banghyunmin/thermal-people-yolov11n](https://huggingface.co/banghyunmin/thermal-people-yolov11n), revision `fbeb08a979248fe3cbb2756f95401da98103956d`. SHA-256: `fdad65ed79d7fd4371e038aa3e43c7ac168addb40f256e316d9f00a78dfbf9a2`. The publisher describes training on 15,303 Roboflow thermal person images and states CC BY 4.0. Ultralytics framework terms also apply.
- `thermal-person.pt`: `model.pt` from [pitangent-ds/YOLOv8-human-detection-thermal](https://huggingface.co/pitangent-ds/YOLOv8-human-detection-thermal), revision `a5d30f1e8e185a54a83cb6288d38c806e303bdd2`. SHA-256: `63181e6c024a8a218020932b516bfb9547130e03e07c8d4671cb2973df038bef`. Publisher license: AGPL-3.0; checkpoint class: `HUMAN`.

## Validation media

Publisher reference: `v2/ADAM/ADAM-Xlarge/test_batch0_gt.jpg` inside the pinned thermal archive. Used to check valid person-box decoding and class mapping, not to claim held-out accuracy. Another inference sanity-check image came from [RescueVision/test.jpg](https://github.com/pandeysarvagya/RescueVision/blob/main/test.jpg). All diagnostic images stay in the ignored `.tools` folder.
