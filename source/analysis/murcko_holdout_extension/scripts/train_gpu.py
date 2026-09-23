"""Resume-safe eight-head training with additional frozen Murcko exclusion."""
import argparse
import json
import os
from types import SimpleNamespace
from pathlib import Path
import torch
import trainer_core as t
import base_trainer as base
from common import PACKAGE, ARMS, TRAINING, expected_jobs, verify_contract, partition, read_frame, write_json
from gpu_preflight import runtime_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', type=Path, default=PACKAGE.parents[1])
    p.add_argument('--worker-rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    args = p.parse_args()
    if not 0 <= args.worker_rank < args.world_size:
        raise ValueError('Worker geometry')
    c = verify_contract()
    run = json.loads((PACKAGE / 'gpu_output/RUN_CONTRACT.json').read_text())
    if run['status'] != 'validated' or run['cpu_manifest_digest'] != c['manifest_digest']:
        raise RuntimeError('Run preflight first')
    threads = int(os.environ.get('TORCH_NUM_THREADS', '4'))
    if threads < 1 or threads != run['torch_num_threads']:
        raise ValueError('TORCH_NUM_THREADS must match preflight')
    torch.set_num_threads(threads)
    cost = dict(ligand=.6, protein=1., c1=1.5, c2=2., c3=8., d1=1.2, d2=1.8, d3=2.5)
    frames = {a: read_frame(a) for a in ARMS}
    sizes = {(a, f): len(partition(frames[a], f)[0]) for a in ARMS for f in range(5)}
    jobs = sorted(expected_jobs(), key=lambda j: (-sizes[j[0], j[3]] * cost[j[1]], j))
    bins, loads = [[] for _ in range(args.world_size)], [0.] * args.world_size
    for j in jobs:
        w = min(range(args.world_size), key=lambda k: (loads[k], k))
        bins[w].append(j)
        loads[w] += sizes[j[0], j[3]] * cost[j[1]]
    mine = bins[args.worker_rank]
    output = PACKAGE / 'gpu_output/benchmark'
    config = SimpleNamespace(**TRAINING)
    pocket = json.loads((PACKAGE / 'data/POCKET_INDICES.json').read_text())
    done = 0
    for arm in ARMS:
        selected = [j for j in mine if j[0] == arm]
        if not selected:
            continue
        frame = runtime_frame(arm, args.project_root.resolve())
        print('Preloading %s; %d assigned fits' % (arm, len(selected)), flush=True)
        caches = base.load_caches(frame, pocket, TRAINING['max_atoms'])
        for a, model, seed, fold in selected:
            report = t.fit_one(base, config, frame, caches, a, 'pfam_murcko', model,
                               seed, fold, output, torch.device('cuda:0'), run)
            assert report['status'] == 'validated'
            done += 1
            progress = dict(status='running' if done < len(mine) else 'validated',
                completed=done, assigned=len(mine), worker=args.worker_rank,
                run_fingerprint=run['run_fingerprint'], last_fit=[a, model, seed, fold])
            write_json(output / 'workers' / ('worker_%d.json' % args.worker_rank), progress)
            print('PROGRESS worker=%d %d/%d' % (args.worker_rank, done, len(mine)), flush=True)
        del caches


if __name__ == '__main__':
    main()
