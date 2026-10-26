"""Verified thermal checkpoints and batched inference over native-resolution crops."""

import gc
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parent


def tile_windows(width, height, tile_size=384, overlap=128):
    """Cover every source pixel, including the right and bottom edges."""
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError('Tile overlap must be smaller than the positive tile size.')

    def starts(length):
        last = max(length - tile_size, 0)
        return sorted(set(range(0, last + 1, tile_size - overlap)) | {last})

    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in starts(height) for x in starts(width)]


def merge_detections(detections, iou_threshold=0.45):
    """Suppress duplicate person boxes after mapping all crops to the full frame."""
    if not detections:
        return np.empty((0, 5), dtype=np.float32)
    boxes = np.concatenate(detections, axis=0).astype(np.float32)
    if not len(boxes):
        return boxes.reshape(0, 5)
    keep = nms(torch.from_numpy(boxes[:, :4].copy()),
               torch.from_numpy(boxes[:, 4].copy()), iou_threshold).numpy()
    return boxes[keep]


def box_iou(first, second):
    first, second = np.asarray(first), np.asarray(second)
    intersection = np.maximum(0, np.minimum(first[2:4], second[2:4]) -
                              np.maximum(first[:2], second[:2])).prod()
    area_first = np.maximum(0, first[2:4] - first[:2]).prod()
    area_second = np.maximum(0, second[2:4] - second[:2]).prod()
    return float(intersection / max(area_first + area_second - intersection, 1e-6))


def corroborate_detections(detections, verification, minimum_iou=0.25):
    """Keep a primary box only when the independent thermal detector agrees."""
    kept = [box for box in detections
            if any(box_iou(box, other) >= minimum_iou for other in verification)]
    return merge_detections([np.asarray(kept, dtype=np.float32).reshape(-1, 5)])


class PersonConfirmation:
    """Require consecutive observations; never draw boxes after they disappear."""
    def __init__(self, frames=3):
        if frames < 1:
            raise ValueError('Confirmation needs at least one frame.')
        self.frames = frames
        self.previous = []

    def update(self, detections):
        remaining = list(self.previous)
        current = []
        confirmed = []
        for box in sorted(detections, key=lambda b: -b[4]):
            overlaps = [box_iou(box, old) for old, _ in remaining]
            match = int(np.argmax(overlaps)) if overlaps else -1
            hits = 1
            if match >= 0 and overlaps[match] >= 0.25:
                _, previous_hits = remaining.pop(match)
                hits += previous_hits
            current.append((box.copy(), min(hits, self.frames)))
            if hits >= self.frames:
                confirmed.append(box)
        self.previous = current
        return np.asarray(confirmed, dtype=np.float32).reshape(-1, 5)


def prepare_detr_inputs(images, image_size, device, half):
    """Normalize on the GPU, matching DETR's processor and bottom/right padding."""
    resized = []
    for image in images:
        height, width = image.shape[:2]
        scale = image_size / max(height, width)
        if scale < 1:
            raise ValueError('The inference input would shrink source pixels. Increase --imgsz.')
        resized.append(cv2.cvtColor(cv2.resize(image, (round(width * scale), round(height * scale))),
                                    cv2.COLOR_BGR2RGB))
    height = max(image.shape[0] for image in resized)
    width = max(image.shape[1] for image in resized)
    batch = np.zeros((len(images), height, width, 3), dtype=np.uint8)
    mask = np.zeros((len(images), height, width), dtype=np.uint8)
    for index, image in enumerate(resized):
        h, w = image.shape[:2]
        batch[index, :h, :w] = image
        mask[index, :h, :w] = 1
    pixels = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).contiguous().float()
    pixel_mask = torch.from_numpy(mask).to(device).long()
    mean = pixels.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = pixels.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    pixels.div_(255).sub_(mean).div_(std)
    pixels.mul_(pixel_mask[:, None])
    return {'pixel_values': pixels.half() if half else pixels, 'pixel_mask': pixel_mask}


class ThermalDetector:
    def __init__(self, weights, device='cuda:0', image_size=640, batch_size=5):
        weights = Path(weights).resolve()
        manifest = json.loads((ROOT / 'models/manifest.json').read_text(encoding='utf-8'))
        if weights.name not in manifest:
            raise ValueError(f'{weights.name} is not a verified thermal-trained checkpoint. See models/manifest.json.')
        self.metadata = manifest[weights.name]
        if not self.metadata.get('thermal_trained'):
            raise ValueError(f'{weights.name} is not marked as thermal-trained in the verified manifest.')
        with weights.open('rb') as handle:
            actual_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual_hash != self.metadata['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {weights.name}')
        self.device = torch.device(device)
        self.half = self.device.type == 'cuda'
        self.image_size = image_size
        self.batch_size = batch_size
        self.name = self.metadata['display_name']
        self.backend = self.metadata['backend']
        self.last_input_count = 0
        if self.backend == 'detr':
            from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
            from safetensors.torch import load_file
            config_path = ROOT / 'models' / self.metadata['config_file']
            if hashlib.sha256(config_path.read_bytes()).hexdigest() != self.metadata['config_sha256']:
                raise ValueError('DETR configuration checksum mismatch.')
            config = DetrConfig.from_json_file(str(config_path))
            config.use_pretrained_backbone = False
            self.model = DetrForObjectDetection(config)
            state = (load_file(str(weights)) if weights.suffix == '.safetensors'
                     else torch.load(weights, map_location='cpu', weights_only=True))
            self.model.load_state_dict(state, strict=True)
            del state
            self.model.eval().to(self.device)
            if self.half:
                self.model.half()
            self.processor = DetrImageProcessor(do_resize=False)
            self.names = {int(key): value for key, value in self.metadata['labels'].items()}
            self.parameters = sum(p.numel() for p in self.model.parameters())
        elif self.backend == 'yolov5':
            from thermal_runtime import prepare_runtime
            runtime = prepare_runtime()
            sys.path.insert(0, str(runtime))
            from models.yolo import Detect
            # This publisher checkpoint is a legacy full PyTorch model. Its hash
            # is verified above before permitting the legacy pickle loader.
            checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
            model = checkpoint.get('ema') or checkpoint['model']
            for module in model.modules():
                if isinstance(module, (torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU,
                                       torch.nn.ReLU6, torch.nn.SiLU)):
                    module.inplace = True
                if isinstance(module, Detect):
                    if not isinstance(module.anchor_grid, list):
                        delattr(module, 'anchor_grid')
                    # Old checkpoints carry cached grids with an older decoding
                    # convention. Rebuild both caches together for this runtime.
                    module.anchor_grid = [torch.empty(0) for _ in range(module.nl)]
                    module.grid = [torch.empty(0) for _ in range(module.nl)]
                    module.inplace = True
                if isinstance(module, torch.nn.Upsample) and not hasattr(module, 'recompute_scale_factor'):
                    module.recompute_scale_factor = None
            self.names = dict(enumerate(model.names)) if isinstance(model.names, list) else model.names
            self.parameters = sum(p.numel() for p in model.parameters())
            self.model = model.float().eval().fuse().to(self.device)
            if self.half:
                self.model.half()
            del checkpoint
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(weights), task='detect')
            self.names = self.model.names
            self.parameters = sum(p.numel() for p in self.model.model.parameters())
        self.person_ids = [int(i) for i, name in self.names.items()
                           if name.lower() in ('person', 'human', 'people')]
        if not self.person_ids:
            raise ValueError(f'Thermal checkpoint has no named person class: {self.names}')

    @torch.inference_mode()
    def predict_batch(self, images, confidence):
        if self.backend == 'detr':
            inputs = prepare_detr_inputs(images, self.image_size, self.device, self.half)
            prediction = self.model(**inputs)
            sizes = torch.tensor([image.shape[:2] for image in images], device=self.device)
            decoded = self.processor.post_process_object_detection(prediction, threshold=confidence,
                                                                   target_sizes=sizes)
            results = []
            for item, (height, width) in zip(decoded, sizes.tolist()):
                valid = torch.zeros_like(item['labels'], dtype=torch.bool)
                for class_id in self.person_ids:
                    valid |= item['labels'] == class_id
                boxes, scores = item['boxes'][valid].float(), item['scores'][valid].float()
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes, scores = boxes[valid], scores[valid]
                keep = nms(boxes, scores, 0.45)
                results.append(torch.cat((boxes[keep], scores[keep, None]), dim=1).cpu().numpy())
            return results
        if self.backend != 'yolov5':
            results = self.model.predict(images, conf=confidence, classes=self.person_ids,
                                         imgsz=self.image_size, device=str(self.device),
                                         quantize=16 if self.half else 32, rect=False, verbose=False)
            return [np.column_stack((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())).astype(np.float32)
                    for r in results]
        prepared = []
        transforms = []
        for image in images:
            height, width = image.shape[:2]
            scale = self.image_size / max(height, width)
            if scale < 1:
                raise ValueError('The inference input would shrink source pixels. Increase --imgsz.')
            resized_width, resized_height = round(width * scale), round(height * scale)
            left = (self.image_size - resized_width) // 2
            top = (self.image_size - resized_height) // 2
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            padded = cv2.copyMakeBorder(resized, top, self.image_size - resized_height - top,
                                       left, self.image_size - resized_width - left,
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))
            prepared.append(padded[:, :, ::-1].transpose(2, 0, 1))
            transforms.append((scale, left, top, width, height))
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(prepared))).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        prediction = self.model(tensor / 255)[0]
        results = []
        for candidates, (scale, left, top, width, height) in zip(prediction, transforms):
            candidates = candidates[candidates[:, 4] >= confidence].float()
            if not len(candidates):
                results.append(np.empty((0, 5), dtype=np.float32))
                continue
            class_scores, class_ids = candidates[:, 5:].max(dim=1)
            scores = candidates[:, 4] * class_scores
            is_person = torch.zeros_like(class_ids, dtype=torch.bool)
            for class_id in self.person_ids:
                is_person |= class_ids == class_id
            valid = is_person & (scores >= confidence)
            candidates, scores = candidates[valid], scores[valid]
            centers, sizes = candidates[:, :2], candidates[:, 2:4]
            boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=1)
            keep = nms(boxes, scores, 0.45)[:100]
            boxes, scores = boxes[keep], scores[keep]
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / scale).clamp(0, width)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / scale).clamp(0, height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores = boxes[valid], scores[valid]
            results.append(torch.cat((boxes, scores[:, None]), dim=1).cpu().numpy())
        return results

    def detect(self, image, confidence=0.5, tiled=True, tile_size=384, overlap=128):
        height, width = image.shape[:2]
        if max(width, height) > self.image_size:
            raise ValueError('The full-frame pass would shrink the image. Increase --imgsz.')
        windows = [(0, 0, width, height)]
        if tiled:
            windows.extend(window for window in tile_windows(width, height, tile_size, overlap)
                           if window != windows[0])
        self.last_input_count = len(windows)
        outputs = []
        index = 0
        while index < len(windows):
            group = windows[index:index + self.batch_size]
            crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in group]
            retry = False
            try:
                predictions = self.predict_batch(crops, confidence)
            except torch.cuda.OutOfMemoryError:
                if self.batch_size <= 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                retry = True
            if retry:
                # Leave the exception scope before releasing its tensor references.
                gc.collect()
                torch.cuda.empty_cache()
                continue
            for prediction, (x1, y1, _, _) in zip(predictions, group):
                prediction[:, [0, 2]] += x1
                prediction[:, [1, 3]] += y1
                outputs.append(prediction)
            index += len(group)
        return merge_detections(outputs)
