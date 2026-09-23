"""Package-integrity tests use temporary fixtures, never production data."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_validator', ROOT/'scripts/validate_release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ReleaseIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        p = root/'code.py'
        p.write_text('x = 1\n')
        return [dict(release_path='code.py', sha256=module.sha(p))]

    def test_valid_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            module.verify_index(root, self.fixture(root))

    def test_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            (root/'code.py').write_text('x = 2\n')
            with self.assertRaises(ValueError):
                module.verify_index(root, rows)

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            (root/'code.py').unlink()
            with self.assertRaises(ValueError):
                module.verify_index(root, rows)

    def test_duplicate_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            with self.assertRaises(ValueError):
                module.verify_index(root, rows+rows)

    def test_parent_path_rejected(self):
        with self.assertRaises(ValueError):
            module.safe_path(ROOT, '../outside')

    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            module.safe_path(ROOT, '/tmp/outside')

    def test_current_extensions_included(self):
        for name in module.REQUIRED_CURRENT:
            self.assertTrue((ROOT/'source/analysis'/name).is_file(), name)

    def test_read_only_real_validation(self):
        before = module.sha(ROOT/'provenance/SOURCE_INDEX.tsv')
        result = module.validate(ROOT, require_manifest=False)
        self.assertEqual(result['status'], 'PASS_SOURCE_PACKAGE_CHECKS')
        self.assertEqual(before, module.sha(ROOT/'provenance/SOURCE_INDEX.tsv'))

if __name__ == '__main__':
    unittest.main()
