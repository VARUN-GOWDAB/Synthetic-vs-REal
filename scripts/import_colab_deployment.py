"""Install a completed Colab export into the portable inference deployment."""
import argparse
import json
import math
from pathlib import Path
import shutil
from colab_training import safe_path, save_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def install(source):
    source = Path(source).resolve()
    manifest = json.loads((source / 'dataset_manifest.json').read_text())
    specification = json.loads((source / 'run_specification.json').read_text())
    status = json.loads((source / 'status.json').read_text())
    if status.get('state') != 'complete' or status.get('experiments') != len(manifest['experiments']):
        raise ValueError('Export is not a completed comparison')
    if sha256(source / 'dataset_manifest.json') != specification['dataset_sha256']:
        raise ValueError('Dataset manifest checksum mismatch')
    if manifest['classes'] != ['worker', 'helmet', 'vest']:
        raise ValueError('Unexpected class mapping')
    names = [e['name'] for e in manifest['experiments']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate experiment name')
    models = []
    holdouts = None
    for experiment in manifest['experiments']:
        name = experiment['name']
        if not name.replace('_', '').isalnum():
            raise ValueError('Invalid experiment name')
        directory = safe_path(source, name)
        result = json.loads((directory / 'test_metrics.json').read_text())
        if result['experiment'] != name or result['sources'] != experiment['sources']:
            raise ValueError(f'Metrics do not match experiment: {name}')
        for split in ('train', 'val', 'test'):
            if result[split + '_images'] != len(experiment['splits'][split]):
                raise ValueError(f'Split count mismatch: {name}')
        shared = (experiment['splits']['val'], experiment['splits']['test'])
        if holdouts is not None and holdouts != shared:
            raise ValueError('Unequal holdouts')
        holdouts = shared
        checkpoint = directory / 'weights/best.pt'
        if sha256(checkpoint) != result['checkpoint_sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
        if not all(math.isfinite(float(v)) for v in result['metrics'].values()):
            raise ValueError(f'Invalid metrics: {name}')
        ratios = {label: round(100 * experiment['sources'].get(key, 0) / result['train_images'], 2)
                  for label, key in [('Real', 'real_safety_500_v4'), ('AI', 'ai_generated_v4'), ('Blender', '3d_rendered')]}
        title = ' / '.join(f'{value:g}% {label}' for label, value in ratios.items() if value)
        models.append({'id': name, 'name': title, 'dataset_version': 4,
                       'checkpoint': f'models/{name}.pt', 'sha256': result['checkpoint_sha256'],
                       'train_images': result['train_images'], 'val_images': result['val_images'],
                       'test_images': result['test_images'], 'ratios': ratios,
                       'metrics': result['metrics'], 'per_class': result['per_class_ap50_95'],
                       'settings': specification['settings'],
                       'dataset': '300-image comparison; shared 7-image validation and 22-image real test sets.',
                       'annotation_status': 'Existing labels; incomplete visual review. Small correlated holdouts; preliminary scores.'})
    # Validate the entire export before changing the active deployment.
    output = ROOT / 'deployment'
    for model in models:
        destination = output / model['checkpoint']
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        shutil.copyfile(source / model['id'] / 'weights/best.pt', temporary)
        if sha256(temporary) != model['sha256']:
            raise ValueError('Checkpoint copy failed')
        temporary.replace(destination)
    provenance = output / 'evaluation'
    provenance.mkdir(exist_ok=True)
    for name in ('dataset_manifest.json', 'run_specification.json', 'environment.json',
                 'initial_weights.json', 'status.json', 'reused_baselines.json', 'comparison_metrics.csv'):
        if (source / name).exists():
            shutil.copy2(source / name, provenance / name)
    for model in models:
        save_json(provenance / (model['id'] + '.json'), json.loads((source / model['id'] / 'test_metrics.json').read_text()))
    counts = {label: sum(r['source'] == key for r in manifest['records'])
              for label, key in [('real', 'real_safety_500_v4'), ('ai', 'ai_generated_v4'), ('blender', '3d_rendered')]}
    save_json(output / 'registry.json', {'version': 1, 'classes': manifest['classes'],
              'dataset_summary': counts, 'models': models, 'status': 'ready',
              'dataset_manifest_sha256': specification['dataset_sha256'],
              'previous_models_archive': 'synthetic_vs_real_cv/archive/models_v2_2026-10-07'})
    print(f'Installed {len(models)} verified models into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    install(parser.parse_args().export)
