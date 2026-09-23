"""Verify the frozen package and explicitly named cached tensors on the GPU host."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from common import PACKAGE, ARMS, TRAINING, SELECTION, sha, digest, write_json, verify_contract, read_frame


def remap(path, root):
    path = str(path)
    for prefix in ('/disk1/11.HS_allostery/', '/disk9/13.Heesu_Allostery/', '/shared_data/11.HS_allostery/'):
        if path.startswith(prefix):
            return str(root / path[len(prefix):])
    return path


def runtime_frame(arm, root):
    d = read_frame(arm)
    for col in ('protein_embedding_path', 'ligand_embedding_path'):
        d[col] = d[col].map(lambda x: remap(x, root))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', type=Path, default=PACKAGE.parents[1])
    args = p.parse_args()
    c = verify_contract()
    root = args.project_root.resolve()
    d = runtime_frame('every_pair', root)
    original = read_frame('every_pair')
    resources = {}
    pairs = {}
    for col in ('protein_embedding_path', 'ligand_embedding_path'):
        pairs.update(zip(original[col], d[col]))
    print('Hashing %d explicitly listed embedding files; no recursive search.' % len(pairs), flush=True)
    for i, (logical, actual) in enumerate(sorted(pairs.items())):
        if not Path(actual).is_file():
            raise FileNotFoundError(actual)
        resources[logical] = dict(sha256=sha(actual), size_bytes=Path(actual).stat().st_size)
        if (i + 1) % 500 == 0:
            print('Input audit %d/%d' % (i + 1, len(pairs)), flush=True)
    previous = json.loads((PACKAGE / 'data/BASELINE_RUN_CONTRACT.json').read_text())
    if resources != previous['embedding_files']:
        raise RuntimeError('Embedding files differ from completed double-held-out baseline. Do not mix representations.')
    import torch
    import trainer_core as t
    import base_trainer as base
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    threads = int(os.environ.get('TORCH_NUM_THREADS', '4'))
    if threads < 1:
        raise ValueError('TORCH_NUM_THREADS must be positive')
    torch.set_num_threads(threads)
    pocket = json.loads((PACKAGE / 'data/POCKET_INDICES.json').read_text())
    sample = d.head(2).copy()
    caches = base.load_caches(sample, pocket, TRAINING['max_atoms'])
    ds = base.PairDataset(sample, caches, TRAINING['max_protein_residues'])
    batch = base.collate_pairs([ds[i] for i in range(len(ds))])
    device = torch.device('cuda:0')
    counts = {}
    for model in t.VALID_MODELS:
        m = t.make_model(model, 256, .30, 4).to(device).eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            logits = m(t.move_batch(batch, device))
        if not torch.isfinite(torch.sigmoid(logits.float())).all():
            raise RuntimeError('Invalid forward pass: ' + model)
        counts[model] = sum(x.numel() for x in m.parameters())
        del m
    if counts != previous['parameter_counts']:
        raise RuntimeError('Architecture parameter counts differ from baseline')
    fingerprint = digest(dict(cpu=c['manifest_digest'], resources=resources,
                               torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                               torch_num_threads=threads))
    output = dict(status='validated', cpu_manifest_digest=c['manifest_digest'],
        run_fingerprint=fingerprint, embedding_files=resources, parameter_counts=counts,
        torch_num_threads=threads, inputs_identical_to_completed_baseline=True,
        versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        split_sha256=c['files']['data/SPLIT_MEMBERSHIP.tsv.gz'],
        cohort_sha256={a: c['files']['data/' + t.ARM_FILES[a]] for a in ARMS},
        training_contract=dict(TRAINING, data_contract_id=c['version'],
            checkpoint_selection=SELECTION,
            training_weight='equal total training weight per protein-label group and then per label; no ligand balancing',
            evaluation_contract='same 4637 test rows; nested family+connectivity+nonempty Murcko exclusion; acyclic separately reported'))
    write_json(PACKAGE / 'gpu_output/RUN_CONTRACT.json', output)
    print(json.dumps({k: output[k] for k in ('status', 'run_fingerprint', 'parameter_counts')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
