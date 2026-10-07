"""Check review integrity, leakage guards, relocation, and safe resume setup."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from colab_training import (extract_bundle, freeze_run, label_counts, save_json,
                            sha256, verify_and_prepare)
from prepare_colab import audited_label_matches, build_manifest
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

    def test_windows_line_endings_do_not_invalidate_inherited_review(self):
        import hashlib
        path = self.root / 'label.txt'
        path.write_bytes(b'0 .5 .5 .2 .2\n')
        old_hash = hashlib.sha256(b'0 .5 .5 .2 .2\r\n').hexdigest()
        self.assertTrue(audited_label_matches(path, old_hash))
        path.write_bytes(b'0 .4 .5 .2 .2\n')
        self.assertFalse(audited_label_matches(path, old_hash))


class ReviewedSelectionTests(unittest.TestCase):
    def test_pilot_excludes_unresolved_and_keeps_balanced_membership(self):
        manifest, report = build_manifest(100, "reviewed_only")
        self.assertEqual(len(manifest['experiments']), 5)
        self.assertEqual(report['holdout_counts'], {'val': 7, 'test': 22})
        self.assertFalse(report['original_full_v4_review_complete'])
        lookup = {r['id']: r for r in manifest['records']}
        selected = set(lookup)
        self.assertFalse(selected.intersection(r['id'] for r in report['exclusions']))
        self.assertTrue(all(len(e['splits']['train']) == 100 for e in manifest['experiments']))
        self.assertTrue(all(e['splits']['test'] == manifest['experiments'][0]['splits']['test']
                            for e in manifest['experiments']))
        self.assertEqual({r['review'] for r in lookup.values() if r['source'] == '3d_rendered'},
                         {'additional_visual_screen'})

    def test_requested_300_uses_existing_labels_without_relabelling(self):
        manifest, report = build_manifest()
        self.assertEqual(manifest['train_images_per_experiment'], 300)
        self.assertEqual(manifest['annotation_review_policy'], 'existing_labels')
        self.assertFalse(manifest['fully_visually_reviewed'])
        self.assertGreater(report['review_provenance_counts']['existing_labels_not_visually_approved'], 0)
        self.assertEqual(len(manifest['records']), 929)
        self.assertEqual([sorted(e['sources'].values()) for e in manifest['experiments']],
                         [[300], [300], [300], [75, 75, 150], [75, 75, 150]])
        selected = {r['id'] for r in manifest['records']}
        self.assertFalse(selected.intersection(r['id'] for r in report['exclusions']))
        for experiment in manifest['experiments']:
            train = experiment['splits']['train']
            self.assertEqual(len(train), 300)
            self.assertEqual(len(set(train)), 300)
            self.assertEqual(experiment['splits']['val'], manifest['experiments'][0]['splits']['val'])
            self.assertEqual(experiment['splits']['test'], manifest['experiments'][0]['splits']['test'])


if __name__ == '__main__':
    unittest.main()
