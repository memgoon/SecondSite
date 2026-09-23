# -*- coding: utf-8 -*-
"""Same audit, restricted to the exact set the published reference metrics use."""
import csv, gzip, statistics
from collections import defaultdict
import numpy as np

R = [r for r in csv.DictReader(gzip.open(
        'gpu_output/reference/reference_predictions.tsv.gz', 'rt'), delimiter='\t')
     if r['weak2020_label'] in ('0', '1')]
ULP = 2.0 ** -11
MOD = ['c2', 'ligand', 'protein']
print('rows %d, targets %d, prevalence %.3f'
      % (len(R), len({r['uniprot'] for r in R}),
         sum(1 for r in R if r['weak2020_label'] == '1') / float(len(R))))

def auroc_ties(pairs):
    pos = [s for s, y in pairs if y]; neg = [s for s, y in pairs if not y]
    if not pos or not neg: return None, None
    pos.sort(); neg.sort()
    import bisect
    win = tie = 0
    for a in pos:
        win += bisect.bisect_left(neg, a)
        tie += bisect.bisect_right(neg, a) - bisect.bisect_left(neg, a)
    n = len(pos) * len(neg)
    return (win + 0.5 * tie) / n, tie / float(n)

def ap(pairs):
    pairs = sorted(pairs, key=lambda x: -x[0])
    tp = 0; s = 0.0; P = sum(1 for _, y in pairs if y)
    for i, (_, y) in enumerate(pairs, 1):
        if y:
            tp += 1; s += tp / float(i)
    return s / P if P else None

print('\n%-9s %10s %10s %14s %8s' % ('model', 'pooled', 'target-', 'tie fraction', 'targets'))
print('%-9s %10s %10s %14s %8s' % ('', 'AUROC', 'macro AUROC', 'within target', 'used'))
for m in MOD:
    allp = [(float(r['p_%s_mean' % m]), r['weak2020_label'] == '1') for r in R]
    pooled, _ = auroc_ties(allp)
    byt = defaultdict(list)
    for r in R:
        byt[r['uniprot']].append((float(r['p_%s_mean' % m]), r['weak2020_label'] == '1'))
    a, t = [], []
    for v in byt.values():
        x, y = auroc_ties(v)
        if x is not None:
            a.append(x); t.append(y)
    print('%-9s %10.4f %10.4f %14.4f %8d'
          % (m, pooled, statistics.mean(a), statistics.mean(t), len(a)))

print('\nscore distribution on this set')
for m in MOD:
    v = np.array([float(r['p_%s_mean' % m]) for r in R])
    print('  p_%-8s range %.4f .. %.4f   frac > 0.99  %.3f   distinct %d'
          % (m, v.min(), v.max(), float((v > 0.99).mean()), len(set(v))))

print('\nwithin-target score span in FP16 ULPs (targets with >=20 rows and both labels)')
for m in MOD:
    byt = defaultdict(list)
    for r in R:
        byt[r['uniprot']].append(float(r['p_%s_mean' % m]))
    sp = [(max(v) - min(v)) / ULP for v in byt.values() if len(v) >= 20]
    print('  %-8s median %8.1f  (%d targets)' % (m, statistics.median(sp), len(sp)))
