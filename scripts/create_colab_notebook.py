"""Generate the uploadable Colab notebook for the currently packaged dataset."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def main():
    checksum = (ROOT / 'colab/synthreal_300_ai_real.sha256').read_text().split()[0]
    cells = []

    def add(kind, text):
        cell = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(text).strip().splitlines(keepends=True)}
        # Preserve a newline between every source line in the notebook format.
        if cell['source'] and not cell['source'][-1].endswith('\n'):
            cell['source'][-1] += '\n'
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cell['id'] = f'cell-{len(cells):02}'
        cells.append(cell)

    add('markdown', '''
    # Synthetic vs. Real — Google Colab GPU training

    This notebook compares eight YOLOv8n models with **300 training images each**
    (five completed models are reused and three new mixtures are trained):
    real only, AI only, rendered only, 50% AI + 25% rendered + 25% real,
    50% real + 25% AI + 25% rendered, 50% AI + 50% real,
    75% real + 25% AI, and 75% AI + 25% real. All use the same **7 real validation images
    and 22 real test images**. Classes: `0 worker`, `1 helmet`, `2 vest`.

    **Existing labels are used without further annotation review**, as requested.
    Previously rejected images and missing files are excluded. Some selected
    training labels remain unresolved in the old review ledger; structural checks
    do not certify their accuracy. The full original v4 review remains incomplete.
    Holdouts are small and contain correlated frames, so results are preliminary.
    Do not compare these scores directly with the bundled v2 model scores.

    **Before running:**
    1. Upload `synthreal_300_ai_real.zip` into a `SynthReal` folder in **My Drive**.
    2. In Colab choose **Runtime → Change runtime type → GPU** (T4 if available).
    3. Run the cells in order. Allow the Google Drive connection when prompted.

    Your computer only needs a browser and internet. Training runs on Colab's GPU.
    Colab GPU availability and runtime limits vary; checkpoints are saved to Drive.
    [Colab FAQ](https://research.google.com/colaboratory/faq.html) ·
    [Ultralytics training and resume documentation](https://docs.ultralytics.com/modes/train/)
    ''')
    add('markdown', '## 1. Check the GPU')
    add('code', '''
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select Runtime > Change runtime type > GPU, reconnect, then rerun this cell.')
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch:', torch.__version__)
    ''')
    add('markdown', '## 2. Connect Google Drive')
    add('code', '''
    from google.colab import drive
    drive.mount('/content/drive')
    ''')
    add('markdown', '''
    ## 3. Copy and verify the dataset
    If your ZIP is somewhere else, change `DRIVE_ARCHIVE` below. Images are copied
    onto the Colab machine for training speed; checkpoints stay on Drive.
    The archive checksum must match this notebook version.
    ''')
    add('code', '''
    from pathlib import Path, PurePosixPath
    import hashlib
    import shutil
    import stat
    import tempfile
    import zipfile

    DRIVE_ARCHIVE = Path('/content/drive/MyDrive/SynthReal/synthreal_300_ai_real.zip')
    EXPECTED_SHA256 = '__CHECKSUM__'
    DATA_ROOT = Path('/content/synthreal_' + EXPECTED_SHA256[:12])
    LOCAL_ARCHIVE = Path('/content/synthreal_300_ai_real.zip')

    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    if not DRIVE_ARCHIVE.is_file():
        raise FileNotFoundError(f'Upload the supplied ZIP to {DRIVE_ARCHIVE}, or update DRIVE_ARCHIVE.')
    if not LOCAL_ARCHIVE.exists() or file_sha256(LOCAL_ARCHIVE) != EXPECTED_SHA256:
        shutil.copyfile(DRIVE_ARCHIVE, LOCAL_ARCHIVE)
    if file_sha256(LOCAL_ARCHIVE) != EXPECTED_SHA256:
        raise ValueError('ZIP checksum mismatch. Use the archive supplied with this notebook.')
    if not DATA_ROOT.exists():
        temporary = Path(tempfile.mkdtemp(prefix='synthreal_extract_', dir='/content'))
        with zipfile.ZipFile(LOCAL_ARCHIVE) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or '..' in path.parts or '\\\\' in member.filename or ':' in member.filename:
                    raise ValueError('Unsafe ZIP member: ' + member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('ZIP symlinks are not supported')
            bundle.extractall(temporary)
        temporary.replace(DATA_ROOT)
    print('Dataset:', DATA_ROOT)
    '''.replace('__CHECKSUM__', checksum))
    add('markdown', '## 4. Install the training library')
    add('code', '''
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           '-r', str(DATA_ROOT / 'requirements-colab.txt')])
    # Keep the CUDA-enabled torch installation supplied by Colab.
    sys.path.insert(0, str(DATA_ROOT))
    from colab_training import verify_and_prepare, run_training
    ''')
    add('markdown', '''
    ## 5. Validate every selected image, label, split, and checksum
    This step creates paths for the current Colab machine. It rejects changed
    annotations, exact duplicate images, overlapping holdouts, and known source
    scene groups crossing splits. It does not certify that every original source
    scene is independent; source video IDs are unavailable.
    ''')
    add('code', '''
    prepared = verify_and_prepare(DATA_ROOT)
    if prepared['manifest']['train_images_per_experiment'] != 300:
        raise ValueError('This notebook requires the 300-image comparison bundle.')
    import pandas as pd
    table = []
    for experiment in prepared['experiments']:
        table.append({'experiment': experiment['name'],
                      **experiment['sources'],
                      'train': len(experiment['splits']['train']),
                      'val': len(experiment['splits']['val']),
                      'test': len(experiment['splits']['test'])})
    display(pd.DataFrame(table).fillna(0))
    print('All selected files and split checks passed. Annotation accuracy has not been certified.')
    ''')
    add('markdown', '''
    ## 6. Train or resume the three new mixtures
    The defaults use 50 epochs, 640-pixel input, batch 8, seed 42, AdamW, and GPU 0.
    The five completed models are included in this bundle. Their source
    membership, image/label hashes, settings, initialization, and checkpoint
    checksums must match before reuse. Their saved test scores are retained.
    Only the three new AI/real mixtures train by default; each starts from the
    same YOLOv8n pretrained weights. The first
    training call downloads those weights. Model selection uses validation only;
    the test set is evaluated after training.

    After a disconnect, reconnect to a GPU and rerun the notebook. Completed
    experiments are skipped and incomplete ones resume from `weights/last.pt`.
    At least one saved epoch is needed to resume. Periodic epoch checkpoints are
    also kept. Do not change data or settings while resuming the same run folder.
    If you deliberately change settings, give `RUN_NAME` a new value and set
    `REUSE_BASELINES = False` to retrain all eight with matching settings. If GPU memory
    is insufficient, use a smaller batch in a **new** run folder for all eight models.
    ''')
    add('code', '''
    EPOCHS = 50
    BATCH = 8
    RUN_NAME = 'ai_real_mixtures_300_seed42_' + EXPECTED_SHA256[:8]
    OUTPUT = Path('/content/drive/MyDrive/SynthReal/runs') / RUN_NAME
    REUSE_BASELINES = True  # Set False in a NEW run folder to retrain all eight.
    results = run_training(prepared, OUTPUT, epochs=EPOCHS, batch=BATCH,
                           reuse_baselines=REUSE_BASELINES)
    print('Saved results and checkpoints:', OUTPUT)
    ''')
    add('markdown', '## 7. Read the comparison')
    add('code', '''
    comparison = pd.read_csv(OUTPUT / 'comparison_metrics.csv')
    display(comparison)
    print('All models have best.pt and test_metrics.json; newly trained models also have last.pt and plots.')
    print('Small, curated holdouts: treat these metrics as preliminary.')
    ''')
    add('markdown', '''
    ## 8. Export the completed models and metrics
    The ZIP stays in Google Drive. It includes all eight best checkpoints, the
    dataset manifest, settings, and measured metrics. Full resumable runs remain
    in the run folder; this smaller export omits optimizer checkpoints.
    ''')
    add('code', '''
    export_path = OUTPUT.parent / (RUN_NAME + '_models_and_metrics.zip')
    paths = [OUTPUT / name for name in ['comparison_metrics.csv', 'run_specification.json',
             'dataset_manifest.json', 'environment.json', 'initial_weights.json', 'status.json']]
    for experiment in prepared['experiments']:
        run = OUTPUT / experiment['name']
        paths.extend([run / 'weights/best.pt', run / 'test_metrics.json'])
    if (OUTPUT / 'reused_baselines.json').is_file():
        paths.append(OUTPUT / 'reused_baselines.json')
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError('Complete training before export. Missing: ' + ', '.join(missing))
    temporary_export = export_path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary_export, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(OUTPUT).as_posix())
    temporary_export.replace(export_path)
    print('Download this file from Google Drive:', export_path)
    ''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
                'metadata': {'colab': {'name': 'Train_SynthReal_on_Colab.ipynb', 'provenance': []},
                             'accelerator': 'GPU',
                             'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}}}
    path = ROOT / 'colab/Train_SynthReal_on_Colab.ipynb'
    path.write_text(json.dumps(notebook, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
