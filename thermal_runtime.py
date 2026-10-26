"""Prepare the pinned upstream YOLOv5 runtime for inference-only startup."""

from pathlib import Path


def prepare_runtime():
    runtime = Path(__file__).resolve().parent / '.tools/yolov5-7.0'
    plotting = runtime / 'utils/plots.py'
    if not plotting.is_file():
        raise RuntimeError('YOLOv5 runtime is missing. Run Setup-PeopleDetection.ps1.')
    source = plotting.read_text(encoding='utf-8')
    # Seaborn eagerly loads the SciPy statistics stack, which is unused during
    # inference. Keep upstream plotting available through a deferred import.
    eager_import = 'import seaborn as sn'
    if eager_import in source:
        deferred_import = (
            'class _LazySeaborn:\n'
            '    def __getattr__(self, name):\n'
            '        import seaborn\n'
            '        return getattr(seaborn, name)\n'
            'sn = _LazySeaborn()'
        )
        plotting.write_text(source.replace(eager_import, deferred_import), encoding='utf-8')
    return runtime
