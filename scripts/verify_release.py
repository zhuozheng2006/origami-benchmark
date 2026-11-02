"""Verify release content without changing or regenerating the manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT/'release-manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Release mismatch: '+name)
    return manifest


if __name__ == '__main__':
    m = verify()
    print('Verified', len(m['sha256']), 'files;', m['version'])
