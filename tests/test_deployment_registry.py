"""Protect deployment-only startup and rejection of retired review writes."""
import json
import os
from http.server import ThreadingHTTPServer
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.dashboard.registry import ROOT, get_registry
from src.dashboard.server import Handler, TOKEN


class DeploymentRegistryTests(unittest.TestCase):
    def test_direct_startup_uses_all_completed_models_without_legacy_mode_switch(self):
        saved = json.loads((ROOT / 'deployment/registry.json').read_text())
        with patch.dict(os.environ, {'SYNTHREAL_DEPLOYMENT': '0'}):
            registry = get_registry()
        self.assertEqual(registry['mode'], 'deployment')
        self.assertEqual([m['id'] for m in registry['models']], [m['id'] for m in saved['models']])
        self.assertEqual(len(registry['models']), 8)
        self.assertTrue(all(m['checkpoint_available'] for m in registry['models']))
        self.assertEqual(registry['queues'], [])
        self.assertFalse(registry['test_review']['available'])

    def test_checkpoint_cannot_escape_deployment_directory(self):
        with patch('src.dashboard.registry.read', return_value={
                'models': [{'checkpoint': '../outside.pt'}]}):
            with self.assertRaisesRegex(ValueError, 'Invalid deployment checkpoint path'):
                get_registry()

    def test_missing_registry_reports_colab_import_requirement(self):
        with patch('src.dashboard.registry.read', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Import a completed Colab export'):
                get_registry()

    def test_retired_review_endpoints_reject_reads_and_authenticated_writes(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/api/review'
            requests = [Request(url), Request(url, data=b'{"index": 0}', headers={
                'Content-Type': 'application/json', 'X-SynthReal-Token': TOKEN})]
            for request in requests:
                with self.subTest(method=request.get_method()):
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(request, timeout=2)
                    self.assertEqual(caught.exception.code, 400)
                    with caught.exception as response:
                        self.assertIn('inference-only mode', json.load(response)['error'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
