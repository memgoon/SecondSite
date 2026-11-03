# SPDX-License-Identifier: MIT
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--dt', type=float, default=0.002)
    ap.add_argument('--out', default='COLVAR_all')
    a = ap.parse_args()
    d = Path(a.dir)
    prog = json.loads((d / 'PROGRESS.json').read_text())
    segs = sorted(prog['segments'], key=lambda x: x['segment'])
    header = None
    rows = []
    used = []
    for s in segs:
        f = d / ('seg_%s.colvar' % s['segment'])
        if not f.exists():
            print('  seg_%s: %s missing, skipped' % (s['segment'], f.name))
            continue
        t0 = s['start_step'] * a.dt
        lim = (s.get('valid_to_step', s['end_step']) - s['start_step']) * a.dt + 1e-06
        n = 0
        for l in f.read_text().splitlines():
            if l.startswith('#! FIELDS'):
                if header is None:
                    header = l
                elif header != l:
                    raise SystemExit('seg_%s has different FIELDS' % s['segment'])
                continue
            if l.startswith('#') or not l.strip():
                continue
            p = l.split()
            lt = float(p[0])
            if lt > lim:
                continue
            p[0] = '%.6f' % (t0 + lt)
            rows.append((t0 + lt, ' '.join(p)))
            n += 1
        used.append('seg_%s[%s] %+.3f ps offset, %d rows (<= %.3f ps local)' % (s['segment'], s.get('status', '?'), t0, n, lim))
    if header is None:
        raise SystemExit('no per-segment COLVAR found')
    rows.sort(key=lambda x: x[0])
    out = []
    last = None
    for (t, l) in rows:
        if last is not None and abs(t - last) < 1e-09:
            continue
        out.append(l)
        last = t
    (d / a.out).write_text(header + '\n' + '\n'.join(out) + '\n')
    for u in used:
        print('  ' + u)
    print('  wrote %s: %d rows, %.3f -> %.3f ps' % (a.out, len(out), float(out[0].split()[0]), float(out[-1].split()[0])))
if __name__ == '__main__':
    main()
