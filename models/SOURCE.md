Model: [pitangent-ds/YOLOv8-human-detection-thermal](https://huggingface.co/pitangent-ds/YOLOv8-human-detection-thermal)

- File: `model.pt`, stored locally as `thermal-person.pt`
- Repository revision: `a5d30f1e8e185a54a83cb6288d38c806e303bdd2`
- SHA-256: `63181e6c024a8a218020932b516bfb9547130e03e07c8d4671cb2973df038bef`
- License stated by publisher: AGPL-3.0
- Class: `HUMAN`

Reference test image: [RescueVision/test.jpg](https://github.com/pandeysarvagya/RescueVision/blob/main/test.jpg), downloaded only into the ignored `.tools` folder for an inference sanity check. This check does not measure accuracy on the connected camera.

## GPU models

- `yolo11l.pt`: [official Ultralytics release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt), [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11). COCO pretrained, about 25 million parameters, AGPL-3.0. This model is not trained specifically on thermal images.
- `thermal-yolo11n.pt`: `best.pt` from [banghyunmin/thermal-people-yolov11n](https://huggingface.co/banghyunmin/thermal-people-yolov11n). The publisher describes training on 15,303 Roboflow thermal person images and states CC BY 4.0. Ultralytics framework terms also apply. Installed as an alternate thermal detector.
