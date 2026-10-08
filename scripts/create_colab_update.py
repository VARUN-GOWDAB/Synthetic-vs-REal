"""Generate a small update ZIP and notebook that reuse the original Colab data."""
from pathlib import Path
import json
import textwrap
import zipfile
from colab_training import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'colab'
    patch = directory / 'synthreal_experiment_update.zip'
    with zipfile.ZipFile(directory / 'synthreal_300_ai_real.zip') as source, zipfile.ZipFile(patch, 'w', zipfile.ZIP_DEFLATED) as out:
        for name in source.namelist():
            if name in ('dataset_manifest.json', 'selection_report.json', 'colab_training.py', 'requirements-colab.txt') or (name.startswith('previous_baselines/') and not name.endswith('.pt')):
                out.writestr(name, source.read(name))
        out.write(ROOT / 'scripts/colab_existing_data.py', 'colab_existing_data.py')
    digest = sha256(patch)
    patch.with_suffix('.sha256').write_text(digest + '  ' + patch.name + '\n')
    nb = json.loads((directory / 'Train_SynthReal_on_Colab.ipynb').read_text())
    setup = textwrap.dedent('''
    from pathlib import Path
    import hashlib
    import shutil
    import stat
    import sys
    import tempfile
    import zipfile

    DRIVE = Path('/content/drive/MyDrive/SynthReal')
    UPDATE_ZIP = DRIVE / 'synthreal_experiment_update.zip'
    EXISTING_DATA = Path('/content/synthreal_5ea8644bd890')
    OLD_DATA_ZIP = DRIVE / 'synthreal_300.zip'
    PREVIOUS_RUN = DRIVE / 'runs/existing_labels_300_seed42_5ea8644b'
    EXPECTED_SHA256 = '__DIGEST__'
    DATA_ROOT = Path('/content/synthreal_update_' + EXPECTED_SHA256[:12])

    if not UPDATE_ZIP.is_file():
        raise FileNotFoundError(f'Upload only the small update ZIP to {UPDATE_ZIP}')
    if hashlib.sha256(UPDATE_ZIP.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Update ZIP and notebook do not match; use the latest pair.')
    if not DATA_ROOT.exists():
        temporary = Path(tempfile.mkdtemp(prefix='synthreal_update_', dir='/content'))
        with zipfile.ZipFile(UPDATE_ZIP) as bundle:
            for member in bundle.infolist():
                p = Path(member.filename)
                if p.is_absolute() or '..' in p.parts or chr(92) in member.filename or ':' in member.filename or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Unsafe update member: ' + member.filename)
            bundle.extractall(temporary)
        temporary.replace(DATA_ROOT)
    sys.path.insert(0, str(DATA_ROOT))
    # Avoid reusing a helper imported by an older notebook in the same session.
    for module in ('colab_training', 'colab_existing_data'):
        sys.modules.pop(module, None)
    from colab_existing_data import prepare_existing_data
    prepare_existing_data(DATA_ROOT, EXISTING_DATA, OLD_DATA_ZIP, PREVIOUS_RUN)
    print('Reused existing dataset and all five completed models:', DATA_ROOT)
    ''').strip().replace('__DIGEST__', digest) + '\n'
    for cell in nb['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'DRIVE_ARCHIVE =' in text:
            text = setup
        elif cell['cell_type'] == 'markdown' and text.startswith('## 3.'):
            text = '''## 3. Reuse your existing dataset and completed models
Upload only `synthreal_experiment_update.zip` to `MyDrive/SynthReal/`.
This cell uses the dataset already in this runtime, or reads the original
`synthreal_300.zip` already in Drive after a runtime reset. It copies the five
completed `best.pt` checkpoints from your previous Drive run and verifies hashes.
No images or models need to be uploaded again. If you moved the original ZIP or
run folder, change `OLD_DATA_ZIP` or `PREVIOUS_RUN` below. Original files stay intact.
'''
        else:
            text = text.replace('Upload `synthreal_300_ai_real.zip`', 'Keep your original dataset ZIP and completed run folder in Drive. Upload only the small `synthreal_experiment_update.zip`')
            text = text.replace("RUN_NAME = 'ai_real_mixtures_300_seed42_'", "RUN_NAME = 'expanded_8_models_300_seed42_'")
        cell['source'] = text.splitlines(keepends=True)
    name = 'Update_Existing_Dataset_on_Colab.ipynb'
    nb['metadata']['colab']['name'] = name
    (directory / name).write_text(json.dumps(nb, indent=2) + '\n')
    print(f'{patch}: {patch.stat().st_size / 1024:.1f} KiB')
    print(directory / name)


if __name__ == '__main__':
    main()
