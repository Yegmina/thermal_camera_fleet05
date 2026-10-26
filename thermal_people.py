"""Live, local person detection for the USB ThermalCamera."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
TITLE = 'Thermal Camera - People Detection'


class Camera:
    def __init__(self, name):
        ffmpeg = next((ROOT / '.tools').glob('**/ffmpeg.exe'))
        self.log = (ROOT / '.tools' / 'detection-camera.log').open('w')
        self.process = subprocess.Popen(
            [str(ffmpeg), '-hide_banner', '-loglevel', 'warning', '-f', 'dshow',
             '-rtbufsize', '64M', '-video_size', '640x512', '-framerate', '50',
             '-i', f'video={name}', '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=self.log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.closed = False
        self.worker = threading.Thread(target=self._read, daemon=True)
        self.worker.start()

    def _read(self):
        frame_bytes = 640 * 512 * 3
        while not self.closed:
            data = bytearray()
            while len(data) < frame_bytes:
                chunk = self.process.stdout.read(frame_bytes - len(data))
                if not chunk:
                    with self.condition:
                        self.closed = True
                        self.condition.notify_all()
                    return
                data.extend(chunk)
            frame = np.frombuffer(data, dtype=np.uint8).reshape(512, 640, 3)
            with self.condition:
                self.frame = frame
                self.sequence += 1
                self.condition.notify_all()

    def latest(self, previous):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence != previous or self.closed, timeout=5
            )
            if self.closed or not ready:
                raise RuntimeError('Camera stopped sending frames. Check the USB connection and detection-camera.log.')
            return self.sequence, self.frame.copy()

    def close(self):
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.worker.join(timeout=2)
        self.process.stdout.close()
        self.log.close()


def prepare(frame, invert=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, [1, 99])
    if high - low >= 3:
        gray = np.clip((gray.astype(np.float32) - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    if invert:
        gray = 255 - gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def write_status(status):
    path = ROOT / '.tools' / 'detection-status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='ThermalCamera')
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'yolo11l.pt')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 (default), cpu, or auto.')
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--vram-gb', type=float, default=8.0,
                        help='GPU budget in GiB, including 1 GiB allowance for the CUDA runtime.')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--rotation', type=int, choices=(0, 90, 180, 270), default=90,
                        help='Counterclockwise rotation to orient this camera upright.')
    parser.add_argument('--duration', type=float, default=0, help='Stop after this many seconds; 0 runs until closed.')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--snapshot', type=Path, help='Save the last annotated frame when exiting.')
    args = parser.parse_args()
    if not 0.05 <= args.confidence <= 0.95:
        parser.error('Confidence must be between 0.05 and 0.95.')
    if args.imgsz < 320 or args.imgsz % 32:
        parser.error('Image size must be at least 320 and a multiple of 32.')
    if args.vram_gb < 2:
        parser.error('GPU budget must be at least 2 GiB.')
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    use_gpu = device.startswith('cuda')
    gpu_name = None
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Start with the .venv-gpu Python environment, or explicitly use --device cpu.')
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
        # PyTorch's allocator is capped below the requested budget, leaving room
        # for the CUDA context and allocations outside that allocator.
        allowance = min(args.vram_gb - 1.0, properties.total_memory / 1024**3 - 1.0)
        torch.cuda.set_per_process_memory_fraction(allowance * 1024**3 / properties.total_memory, device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    model = YOLO(str(args.model), task='detect')
    person_ids = [int(i) for i, label in model.names.items() if label.lower() in ('person', 'human', 'people')]
    if not person_ids:
        raise RuntimeError(f'Model has no person class: {model.names}')
    print(f'Model: {args.model.name}; device: {device}; GPU: {gpu_name}; input: {args.imgsz}; budget: {args.vram_gb} GiB', flush=True)
    predict_options = dict(conf=args.confidence, classes=person_ids, imgsz=args.imgsz,
                           device=device, half=use_gpu, verbose=False)
    model.predict(np.zeros((512, 640, 3), dtype=np.uint8), **predict_options)
    if use_gpu:
        torch.cuda.synchronize(device)
    camera = Camera(args.camera)
    sequence = 0
    count = 0
    frames = 0
    fps = 0.0
    confidence = args.confidence
    invert = False
    rotation = args.rotation
    fullscreen = False
    annotated = None
    status = {'model': args.model.name, 'device': device, 'gpu_name': gpu_name,
              'vram_budget_gib': args.vram_gb, 'half_precision': use_gpu,
              'input_size': args.imgsz}
    started = time.monotonic()
    last_status = 0
    if not args.headless:
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, 768 if rotation % 180 else 960, 960 if rotation % 180 else 768)
    try:
        while not args.duration or time.monotonic() - started < args.duration:
            tick = time.monotonic()
            sequence, frame = camera.latest(sequence)
            if rotation:
                frame = cv2.rotate(frame, {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                                          180: cv2.ROTATE_180,
                                          270: cv2.ROTATE_90_CLOCKWISE}[rotation])
            image = prepare(frame, invert)
            height, width = image.shape[:2]
            predict_options['conf'] = confidence
            result = model.predict(image, **predict_options)[0]
            boxes = result.boxes
            count = len(boxes)
            annotated = image.copy()
            detections = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0])
                detections.append({'box': [x1, y1, x2, y2], 'confidence': round(score, 3)})
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 240, 80), 2)
                cv2.putText(annotated, f'Person {score:.0%}', (x1, max(48, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 80), 2)
            elapsed = max(time.monotonic() - tick, 0.001)
            fps = 1 / elapsed if not fps else fps * 0.9 + 0.1 / elapsed
            cv2.rectangle(annotated, (0, 0), (width, 56), (20, 20, 20), -1)
            cv2.putText(annotated, f'People: {count}   |   {fps:.1f} FPS   |   {"GPU" if use_gpu else "CPU"}',
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (0, 240, 80), 1, cv2.LINE_AA)
            cv2.putText(annotated, f'{args.model.stem}   |   Confidence: {confidence:.0%}',
                        (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(annotated, (0, height - 25), (width, height), (20, 20, 20), -1)
            cv2.putText(annotated, 'Q/Esc: close  F: fullscreen  +/-: sensitivity  I: invert  R: rotate',
                        (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
            frames += 1
            if time.monotonic() - last_status >= 1:
                memory = {} if not use_gpu else {
                    'allocated_mib': round(torch.cuda.memory_allocated(device) / 1024**2, 1),
                    'reserved_mib': round(torch.cuda.memory_reserved(device) / 1024**2, 1),
                    'peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
                    'peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
                }
                status.update({'running': True, 'timestamp': time.time(), 'people': count,
                              'fps': round(fps, 1), 'processed_frames': frames, 'confidence': confidence,
                              'device': device, 'gpu_name': gpu_name, 'model': args.model.name,
                              'input_size': args.imgsz, 'half_precision': use_gpu,
                              'vram_budget_gib': args.vram_gb, 'cuda_memory': memory,
                              'rotation': rotation, 'detections': detections})
                write_status(status)
                last_status = time.monotonic()
            if not args.headless:
                cv2.imshow(TITLE, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                try:
                    visible = cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    break
                if visible < 1:
                    break
                if key == ord('f'):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(TITLE, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                elif key in (ord('+'), ord('=')):
                    confidence = max(0.05, round(confidence - 0.05, 2))
                elif key == ord('-'):
                    confidence = min(0.95, round(confidence + 0.05, 2))
                elif key == ord('i'):
                    invert = not invert
                elif key == ord('r'):
                    rotation = (rotation + 90) % 360
    finally:
        camera.close()
        cv2.destroyAllWindows()
        if args.snapshot and annotated is not None:
            if not cv2.imwrite(str(args.snapshot), annotated):
                raise RuntimeError(f'Could not save snapshot: {args.snapshot}')
        status.update({'running': False, 'timestamp': time.time(), 'people': count,
                       'processed_frames': frames, 'fps': round(fps, 1)})
        write_status(status)
        print(f'Stopped after {frames} frames; {fps:.1f} FPS; last person count: {count}', flush=True)


if __name__ == '__main__':
    main()
