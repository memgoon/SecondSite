"""Stage 6 - write the package checksum manifest.

Cache payloads are hashed as well, so a rerun that silently picks up a different
ChEBI or KEGG release fails verification instead of producing different numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SKIP = {'__pycache__', '.tmp'}


def main():
    targets = []
    for root, dirs, files in os.walk(C.PKG):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in sorted(files):
            if f.endswith('.tmp') or f == 'CHECKSUMS.sha256':
                continue
            p = os.path.join(root, f)
            targets.append((C.sha256(p), os.path.relpath(p, C.PKG)))
    targets.sort(key=lambda t: t[1])
    out = os.path.join(C.PKG, 'manifests', 'CHECKSUMS.sha256')
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        for h, rel in targets:
            fh.write('%s  %s\n' % (h, rel))
    os.replace(tmp, out)
    print('PASS checksum_targets=%d' % len(targets))


if __name__ == '__main__':
    main()
