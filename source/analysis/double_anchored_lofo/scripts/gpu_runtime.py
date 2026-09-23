"""Fixed-epoch training; no validation, no test-time checkpoint selection."""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import base_trainer as base
import model_definitions as models
from common import *
from queue_jobs import initialize, claim, finish


def load_inputs():
    frame = read_data()
    root = PACKAGE.parents[1]
    for field in ('ligand_embedding_path', 'protein_embedding_path'):
        frame[field] = frame[field].map(lambda p: local_path(p, root))
    pockets = json.loads((PACKAGE / 'data/POCKET_INDICES.json').read_text())
    caches = base.load_caches(frame, pockets, TRAINING['max_atoms'])
    if any(t.shape[0] > TRAINING['max_protein_residues'] for t in caches[1].values()):
        raise ValueError('Unexpected protein truncation; review contract before training')
    dataset = base.PairDataset(frame, caches, TRAINING['max_protein_residues'])
    samples = {str(frame.iloc[i].main_row_id): dataset[i] for i in range(len(dataset))}
    return frame, samples


def make_model(name):
    return models.make_model(name, TRAINING['hidden_dim'], TRAINING['dropout'], TRAINING['heads'])


def gpu_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}


def preflight():
    contract = verify_contract()
    original = read_data()
    paths = sorted(set(original.ligand_embedding_path) | set(original.protein_embedding_path))
    input_hashes = {p: sha(local_path(p, PACKAGE.parents[1])) for p in paths}
    frame, samples = load_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda:0')
    batch = gpu_batch(base.collate_pairs([samples[str(frame.iloc[0].main_row_id)]]), device)
    parameter_counts = {}
    for name in MODELS:
        model = make_model(name).to(device).eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(batch)
        probability = torch.sigmoid(logits.float())
        if not torch.isfinite(probability).all():
            raise ValueError('Nonfinite preflight output')
        parameter_counts[name] = sum(p.numel() for p in model.parameters())
        del model
    core = dict(cpu_manifest=contract['manifest_digest'], input_hashes=input_hashes,
                training=TRAINING, torch_version=str(torch.__version__), numpy_version=np.__version__,
                pandas_version=pd.__version__, version=VERSION)
    run = dict(core, run_fingerprint=digest(core), status='validated', parameter_counts=parameter_counts)
    path = PACKAGE / 'gpu_output/RUN_CONTRACT.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['run_fingerprint'] != run['run_fingerprint']:
            raise RuntimeError('Existing run fingerprint differs. Preserve old output; do not mix runs.')
    write_json(path, run)
    initialize(run)
    print(json.dumps(dict(status='validated', fits=len(jobs()), parameters=parameter_counts), indent=2), flush=True)


def predict(model, name, test, samples, device):
    # Actual single-input identity, not score rounding. Avoid padding-dependent AMP noise.
    test = test.copy()
    test['_input_key'] = input_identity(name, test)
    unique = test.drop_duplicates('_input_key')
    batch_size = 1 if name in ('protein', 'ligand') else TRAINING['c3_eval_batch_size' if name == 'c3' else 'eval_batch_size']
    loader = DataLoader([samples[str(k)] for k in unique.main_row_id], batch_size=batch_size,
                        collate_fn=base.collate_pairs, num_workers=0)
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast():
                logits = model(gpu_batch(batch, device))
            values.extend(torch.sigmoid(logits.float()).cpu().numpy().reshape(-1).tolist())
    mapping = dict(zip(unique._input_key, values))
    test['p_allosteric'] = test._input_key.map(mapping)
    if not np.isfinite(test.p_allosteric.to_numpy()).all() or not test.p_allosteric.between(0, 1).all():
        raise ValueError('Invalid final predictions')
    return test.drop(columns=['_input_key'])


def train_fit(job, frame, samples, run, worker):
    start = time.time()
    write_json(PACKAGE / ('gpu_output/progress/worker%s.json' % worker), dict(identity(job), epoch=0, updated=start))
    regime, name, seed, fold = job
    train, test, excluded = partition(frame, regime, fold)
    base.seed_everything(seed)
    device = torch.device('cuda:0')
    model = make_model(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING['lr'], weight_decay=TRAINING['weight_decay'])
    scaler = torch.cuda.amp.GradScaler()
    weights = base.class_and_protein_balanced_weights(train)
    data = [dict(samples[str(k)], weight=float(w)) for k, w in zip(train.main_row_id, weights)]
    batch_size = TRAINING['c3_batch_size' if name == 'c3' else 'batch_size']
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=generator,
                        collate_fn=base.collate_pairs, num_workers=0, pin_memory=True)
    history = []
    directory = fit_dir(job)
    directory.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, TRAINING['epochs'] + 1):
        model.train()
        losses = []
        for batch in loader:
            tensors = gpu_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(tensors)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, tensors['label'], reduction='none') * tensors['weight']).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            losses.append((float(loss.detach().cpu()), len(tensors['label'])))
        avg = sum(v*n for v,n in losses)/sum(n for _,n in losses)
        history.append(dict(epoch=epoch, training_loss=avg, elapsed_seconds=time.time()-start))
        write_json(PACKAGE / ('gpu_output/progress/worker%s.json' % worker), dict(identity(job), epoch=epoch, updated=time.time()))
        print('%s %s seed=%s fold=%s epoch=%s loss=%.6f' % (regime,name,seed,fold,epoch,avg), flush=True)
    # Test is first evaluated only after all 25 training epochs are complete.
    output = predict(model, name, test, samples, device)
    for k,v in identity(job).items():
        output[k] = v
    # Restore frozen path strings for auditable metadata independent of mount point.
    frozen = read_data().set_index('main_row_id')
    for k in ('ligand_embedding_path', 'protein_embedding_path'):
        output[k] = output.main_row_id.map(frozen[k])
    torch.save(dict(state_dict=model.cpu().state_dict(), identity=identity(job), training=TRAINING,
                    run_fingerprint=run['run_fingerprint'], model_version=models.MODEL_VERSION), str(directory / 'final.pt.tmp'))
    os.replace(str(directory / 'final.pt.tmp'), str(directory / 'final.pt'))
    output.to_csv(directory / 'predictions.tsv.gz', sep='\t', index=False)
    pd.DataFrame(history).to_csv(directory / 'history.tsv', sep='\t', index=False)
    report = dict(identity(job), status='validated', training=TRAINING, final_epoch=25, validation_rows=0,
                  train_rows=len(train), test_rows=len(test), excluded_rows=len(excluded),
                  held_out_family=families(frame)[fold], model_version=models.MODEL_VERSION,
                  run_fingerprint=run['run_fingerprint'], elapsed_seconds=time.time()-start,
                  cpu_threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(0),
                  sha256={p:sha(directory/p) for p in ('final.pt','predictions.tsv.gz','history.tsv')})
    write_json(directory / 'FIT_REPORT.json', report)
    print('FIT_COMPLETE '+json.dumps(identity(job)), flush=True)
    del model, optimizer, scaler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['preflight', 'worker'])
    p.add_argument('--worker', default='0')
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get('CPU_THREADS_PER_GPU', '6')))
    torch.set_num_interop_threads(1)
    if args.mode == 'preflight':
        preflight()
        return
    verify_contract()
    run = json.loads((PACKAGE / 'gpu_output/RUN_CONTRACT.json').read_text())
    frame, samples = load_inputs()
    while True:
        job = claim(args.worker)
        if job is None:
            break
        try:
            train_fit(job, frame, samples, run, args.worker)
            finish(job, 'done')
        except BaseException:
            finish(job, 'failed')
            raise


if __name__ == '__main__':
    main()
