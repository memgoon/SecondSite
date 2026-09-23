"""Reject traversal and links before extracting a trusted expected-output archive."""
import sys
import tarfile
from pathlib import Path, PurePosixPath
from common import PACKAGE

with tarfile.open(sys.argv[1], 'r:gz') as archive:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if (path.is_absolute() or '..' in path.parts or member.issym() or member.islnk()
                or not (member.isfile() or member.isdir())
                or path.parts[:3] != ('analysis', 'murcko_holdout_extension', 'gpu_output')):
            raise RuntimeError('Unsafe or unexpected archive entry: ' + member.name)
    archive.extractall(str(PACKAGE.parents[1]))
