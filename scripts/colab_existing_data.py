"""Reuse existing images and completed checkpoints without another large upload."""
from pathlib import Path
import json
import shutil
import zipfile

from colab_training import safe_path, sha256


def copy_verified(source, destination, checksum):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f'Missing existing file: {source}')
    if sha256(source) != checksum:
        raise ValueError(f'Existing file checksum mismatch: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.copying')
    shutil.copyfile(source, temporary)
    if sha256(temporary) != checksum:
        temporary.unlink()
        raise ValueError(f'Copy checksum mismatch: {source}')
    temporary.replace(destination)


def prepare_existing_data(root, existing_root, existing_archive, previous_run):
    root, existing_root = Path(root), Path(existing_root)
    existing_archive, previous_run = Path(existing_archive), Path(previous_run)
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    files = [(r[field], r[field + '_sha256']) for r in manifest['records'] for field in ('image', 'label')]
    missing = []
    for relative, checksum in files:
        destination = safe_path(root, relative)
        if destination.exists():
            if sha256(destination) != checksum:
                raise ValueError(f'Prepared file changed: {destination}')
        elif safe_path(existing_root, relative).is_file():
            copy_verified(safe_path(existing_root, relative), destination, checksum)
        else:
            missing.append((relative, checksum))
    if missing:
        if not existing_archive.is_file():
            raise FileNotFoundError(f'Existing dataset not found. Set EXISTING_DATA or OLD_DATA_ZIP. Expected ZIP: {existing_archive}')
        # Read only explicit manifest paths, never execute or extract old helper code.
        with zipfile.ZipFile(existing_archive) as archive:
            for relative, checksum in missing:
                destination = safe_path(root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + '.copying')
                with archive.open(relative) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
                if sha256(temporary) != checksum:
                    temporary.unlink()
                    raise ValueError(f'Old dataset contents do not match: {relative}')
                temporary.replace(destination)
    previous = root / 'previous_baselines'
    for result_path in sorted(previous.glob('*/test_metrics.json')):
        result = json.loads(result_path.read_text())
        name = result_path.parent.name
        destination = result_path.parent / 'weights/best.pt'
        if destination.exists():
            if sha256(destination) != result['checkpoint_sha256']:
                raise ValueError(f'Prepared checkpoint changed: {name}')
        else:
            source = previous_run / name / 'weights/best.pt'
            if not source.is_file():
                raise FileNotFoundError(f'Set PREVIOUS_RUN to your original completed run folder. Missing: {source}')
            copy_verified(source, destination, result['checkpoint_sha256'])
    return root
