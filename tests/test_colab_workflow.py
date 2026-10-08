"""Check frozen Colab data integrity, leakage guards, baseline reuse and safe resume."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from colab_training import (extract_bundle, freeze_run, label_counts, save_json,
                            sha256, verify_and_prepare, reuse_completed_baselines)
ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / 'results/colab_baselines'
from PIL import Image


class ColabDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        records = []
        for i, split in enumerate(('train', 'val', 'test')):
            folder = self.root / 'data/real_safety_500_v4'
            (folder / 'images').mkdir(parents=True, exist_ok=True)
            (folder / 'labels').mkdir(exist_ok=True)
            image = folder / 'images' / f'{split}.png'
            label = folder / 'labels' / f'{split}.txt'
            Image.new('RGB', (12, 12), (i * 60, 15, 50)).save(image)
            label.write_text('0 0.5 0.5 0.5 0.5\n')
            records.append({'id': split, 'source': 'real_safety_500_v4',
                            'image': image.relative_to(self.root).as_posix(),
                            'label': label.relative_to(self.root).as_posix(),
                            'image_sha256': sha256(image), 'label_sha256': sha256(label),
                            'objects': {'0': 1, '1': 0, '2': 0},
                            'original_split': split, 'scene_group': str(i)})
        self.manifest = {'ready_for_training': True, 'classes': ['worker', 'helmet', 'vest'],
                         'train_images_per_experiment': 1, 'records': records,
                         'experiments': [{'name': 'real_test',
                                          'sources': {'real_safety_500_v4': 1},
                                          'splits': {'train': ['train'], 'val': ['val'], 'test': ['test']}}]}
        self.write()

    def write(self):
        save_json(self.root / 'dataset_manifest.json', self.manifest)

    def test_relocation_builds_valid_local_yolo_paths(self):
        result = verify_and_prepare(self.root)
        path = Path(result['experiments'][0]['data'])
        self.assertTrue(path.exists())
        self.assertEqual(Path(path.parent.joinpath('train.txt').read_text().strip()),
                         self.root / self.manifest['records'][0]['image'])

    def test_changed_label_is_rejected_even_if_still_valid_yolo(self):
        (self.root / self.manifest['records'][0]['label']).write_text('0 0.4 0.5 0.5 0.5\n')
        with self.assertRaisesRegex(ValueError, 'changed since dataset freeze'):
            verify_and_prepare(self.root)

    def test_holdout_in_training_is_rejected(self):
        self.manifest['experiments'][0]['splits']['train'] = ['test']
        self.write()
        with self.assertRaisesRegex(ValueError, 'Holdout image'):
            verify_and_prepare(self.root)

    def test_source_scene_overlap_is_rejected(self):
        self.manifest['records'][0]['scene_group'] = self.manifest['records'][2]['scene_group']
        self.write()
        with self.assertRaisesRegex(ValueError, 'scene group'):
            verify_and_prepare(self.root)

    def test_duplicate_image_bytes_are_rejected(self):
        one, two = self.manifest['records'][:2]
        (self.root / two['image']).write_bytes((self.root / one['image']).read_bytes())
        two['image_sha256'] = one['image_sha256']
        self.write()
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            verify_and_prepare(self.root)

    def test_unequal_holdouts_are_rejected(self):
        other = copy.deepcopy(self.manifest['experiments'][0])
        other['name'] = 'other'
        other['splits']['val'], other['splits']['test'] = other['splits']['test'], other['splits']['val']
        self.manifest['experiments'].append(other)
        self.write()
        with self.assertRaises(ValueError):
            verify_and_prepare(self.root)

    def test_invalid_labels_are_rejected(self):
        for text in ('0 nan .5 .2 .2', '3 .5 .5 .2 .2', '0 .9 .5 .4 .2',
                     '0 .5 .5 0 .2', '0 .5 .5 .2 .2\n0 .5 .5 .2 .2'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                label_counts(text)

    def test_unsafe_zip_cannot_write_outside_destination(self):
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('../escape.txt', 'bad')
        with self.assertRaises(ValueError):
            extract_bundle(archive, self.root / 'extracted')
        self.assertFalse((self.root / 'escape.txt').exists())

    def test_changed_run_settings_cannot_resume(self):
        output = self.root / 'runs'
        freeze_run(output, {'epochs': 50, 'data': 'one'})
        freeze_run(output, {'epochs': 50, 'data': 'one'})
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            freeze_run(output, {'epochs': 51, 'data': 'one'})


class BaselineReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'deployment/evaluation/dataset_manifest.json').read_text())
        cls.spec = json.loads((BASELINES / 'run_specification.json').read_text())
        cls.initial = json.loads((BASELINES / 'initial_weights.json').read_text())['sha256']

    def test_reuses_all_five_completed_models_and_keeps_original_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = reuse_completed_baselines({'manifest': self.manifest}, tmp, BASELINES, self.spec, self.initial)
            self.assertEqual(len(names), 5)
            self.assertEqual(sum('mixed' in name for name in names), 2)
            for name in names:
                self.assertEqual((Path(tmp) / name / 'test_metrics.json').read_bytes(),
                                 (BASELINES / name / 'test_metrics.json').read_bytes())
            self.assertEqual(names, reuse_completed_baselines({'manifest': self.manifest}, tmp, BASELINES, self.spec, self.initial))
            for experiment in self.manifest['experiments'][5:]:
                self.assertNotIn('3d_rendered', experiment['sources'])
                self.assertFalse((Path(tmp) / experiment['name']).exists())

    def test_rejects_changed_baseline_membership_before_copying(self):
        changed = copy.deepcopy(self.manifest)
        changed['experiments'][0]['splits']['train'].reverse()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'membership changed'):
                reuse_completed_baselines({'manifest': changed}, tmp, BASELINES, self.spec, self.initial)
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_rejects_changed_settings_or_initialization(self):
        changed = copy.deepcopy(self.spec)
        changed['settings']['epochs'] = 100
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                reuse_completed_baselines({'manifest': self.manifest}, tmp, BASELINES, changed, self.initial)
            with self.assertRaisesRegex(ValueError, 'initialization differs'):
                reuse_completed_baselines({'manifest': self.manifest}, tmp, BASELINES, self.spec, 'wrong')


if __name__ == '__main__':
    unittest.main()
