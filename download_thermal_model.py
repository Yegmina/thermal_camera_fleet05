"""Download the pinned thermal YOLOv5x checkpoint without the entire model archive."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parent
NAME = 'thermal-yolov5x-adam.pt'
OFFSET = 295883535
COMPRESSED_SIZE = 161521518
UNCOMPRESSED_SIZE = 177550789
CRC32 = 387802185
URL = ('https://huggingface.co/MAli-Farooq/Thermal_Object_Detection/resolve/'
       '0df98a0ed9e200be78a313f5f74d347a90a93a3c/Heliaus-Thermal-SGD-ADAM.zip')


def main():
    expected_hash = json.loads((ROOT / 'models/manifest.json').read_text())[NAME]['sha256']
    destination = ROOT / 'models' / NAME
    if destination.is_file():
        with destination.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() == expected_hash:
                print('Thermal YOLOv5x checkpoint checksum verified.')
                return
    destination.parent.mkdir(exist_ok=True)
    (ROOT / '.tools').mkdir(exist_ok=True)
    temporary = ROOT / '.tools/thermal-yolov5x.download'
    print('Downloading the thermal-trained YOLOv5x checkpoint (about 154 MiB)...', flush=True)
    subprocess.run(['curl.exe', '--location', '--fail', '--retry', '3', '--silent', '--show-error',
                    '--range', f'{OFFSET}-{OFFSET + COMPRESSED_SIZE + 4096}',
                    '--output', str(temporary), URL], check=True)
    data = temporary.read_bytes()
    if len(data) != COMPRESSED_SIZE + 4097:
        raise RuntimeError('The model archive server did not return the requested byte range.')
    header = struct.unpack('<4s5H3I2H', data[:30])
    if header[0] != b'PK\x03\x04' or header[3] != 8:
        raise RuntimeError('Unexpected thermal archive member header.')
    start = 30 + header[-2] + header[-1]
    decoded = zlib.decompress(data[start:start + COMPRESSED_SIZE], -15)
    if len(decoded) != UNCOMPRESSED_SIZE or zlib.crc32(decoded) != CRC32:
        raise RuntimeError('The thermal model archive member failed its integrity check.')
    if hashlib.sha256(decoded).hexdigest() != expected_hash:
        raise RuntimeError('The thermal checkpoint failed its SHA-256 check.')
    staged = destination.with_suffix('.download')
    staged.write_bytes(decoded)
    staged.replace(destination)
    temporary.unlink()
    print(f'Saved verified thermal checkpoint: {destination.name}', flush=True)


if __name__ == '__main__':
    main()
