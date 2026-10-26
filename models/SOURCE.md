ï»¿# Thermal checkpoint sources

## Default: thermal DETR ResNet-50

- Publisher: [faldeus0092/detr-finetuned-thermal-dogs-and-people](https://huggingface.co/faldeus0092/detr-finetuned-thermal-dogs-and-people).
- Pinned revision: `f7c9ff88dfb256c9fdb36e292046439c7f97e21c`.
- Selected file: `pytorch_model.bin`, 166,625,621 bytes, saved as `thermal-detr-r50.bin`.
- SHA-256: `8c10905ad0d26d046eddb664da6cec504fb2ba2c2e7bd8d2306bc926a99cf909`.
- Actual loaded parameter count: 41,502,152. The checkpoint state dictionary loads strictly with PyTorch `weights_only=True` after checksum verification.
- The model card explicitly identifies the training data as Roboflow Thermal Dogs and People, with 142 training images, 41 evaluation images, and 68 epochs. That is a small thermal fine-tuning dataset; camera-specific generalization is unproven.
- The pinned [training notebook](https://huggingface.co/faldeus0092/detr-finetuned-thermal-dogs-and-people/blob/f7c9ff88dfb256c9fdb36e292046439c7f97e21c/detr-finetuned-thermal-dogs-and-people.ipynb) explicitly defines `id2label = {0: "dogs-person", 1: "dog", 2: "person"}`. Only class 2 is used. The saved config has anonymous label names, so the notebook's explicit mapping is recorded in the manifest.
- Original configuration: `thermal-detr-r50-config.json`; SHA-256 `aa86d4795ec53a3b389ad9d0dfd5c51fb6ad9b1c67e97d2812660f5381171157`. Original normalization/resize reference: `thermal-detr-r50-processor.json` (shortest edge 800, maximum 1333). Our default uses maximum edge 640 and source-native overlapping crops, preserving all sensor pixels. It is not a claim that the model was trained at our inference resolution.
- Transformers uses Apache-2.0. The fine-tuned checkpoint's model card does not declare a license and describes its intended purpose as learning. The independent thermal YOLO verifier uses AGPL-3.0.

The inference backend uses Transformers 4.57.6 and the checkpoint's native Transformers ResNet backbone, with automatic pretrained-backbone downloads disabled. GPU normalization is checked against `DetrImageProcessor`; the official `post_process_object_detection` maps boxes to original image dimensions.

The default independent verifier is `thermal-person.pt` below. Both models process the full frame plus four overlapping crops; their final boxes must overlap by at least 0.25 IoU. Three consecutive observations are required. These filters trade recall and confirmation delay for fewer unsupported detections; persistent false positives can still pass.

Two alternatives were tested and excluded: the aerial thermal RT-DETR-L from `Kiuyha/rtdetrv2-human-detection-thermal-uav` missed the ground-level reference people, and `Inishds/detr-resnet-101-dc5-FLIR-data-set` did not document whether its FLIR training data used thermal or visible frames. Neither is an approved manifest entry or a setup download.

## Legacy alternative: thermal YOLOv5x

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
