# SPDX-License-Identifier: MIT
import argparse, json, re, sys
from pathlib import Path

def parse_schedule(path):
    steps = []
    for l in Path(path).read_text().splitlines():
        m = re.match('\\s*STEP(\\d+)\\s*=\\s*(\\d+)\\s+AT\\1\\s*=\\s*(\\S+)\\s+KAPPA\\1\\s*=\\s*(\\S+)', l)
        if m:
            steps.append((int(m.group(2)), [float(x) for x in m.group(3).split(',')], [float(x) for x in m.group(4).split(',')]))
        a = re.match('\\s*ARG\\s*=\\s*(\\S+)', l)
    args = None
    for l in Path(path).read_text().splitlines():
        m = re.match('\\s*ARG\\s*=\\s*(\\S+)\\s*$', l)
        if m and ',' in m.group(1):
            args = m.group(1).split(',')
    if not steps:
        raise SystemExit('no schedule found in ' + str(path))
    if not args:
        raise SystemExit('no ARG= list found in ' + str(path))
    steps.sort(key=lambda x: x[0])
    return (args, steps)

def at_step(steps, S):
    if S <= steps[0][0]:
        return (steps[0][1], steps[0][2])
    if S >= steps[-1][0]:
        return (steps[-1][1], steps[-1][2])
    for ((s0, a0, k0), (s1, a1, k1)) in zip(steps, steps[1:]):
        if s0 <= S <= s1:
            f = (S - s0) / (s1 - s0)
            return ([x + f * (y - x) for (x, y) in zip(a0, a1)], [x + f * (y - x) for (x, y) in zip(k0, k1)])
    raise SystemExit('step %d outside the schedule' % S)

def read_colvar(path):
    fields = None
    rows = []
    for l in Path(path).read_text().splitlines():
        if l.startswith('#! FIELDS'):
            fields = l.split()[2:]
            continue
        if l.startswith('#') or not l.strip():
            continue
        rows.append([float(x) for x in l.split()])
    if fields is None:
        raise SystemExit('no "#! FIELDS" header in ' + str(path))
    return (fields, rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--colvar', default='COLVAR_all', help='the stitched absolute-time file (h3e_stitch_colvar.py output)')
    ap.add_argument('--dt', type=float, default=0.002)
    ap.add_argument('--cntr-tol', type=float, default=0.0001)
    ap.add_argument('--kappa-tol', type=float, default=1.0)
    a = ap.parse_args()
    d = Path(a.dir)
    (args, steps) = parse_schedule(d / 'plumed_original.dat')
    (fields, rows) = read_colvar(d / a.colvar)
    if not rows:
        raise SystemExit('COLVAR holds no data rows')
    idx = {f: i for (i, f) in enumerate(fields)}

    def find(suffix, cv):
        hits = [f for f in fields if f.endswith('.%s_%s' % (cv, suffix))]
        if len(hits) != 1:
            raise SystemExit('expected exactly one *.%s_%s column, found %s' % (cv, suffix, hits))
        return idx[hits[0]]
    cols = {cv: (find('cntr', cv), find('kappa', cv)) for cv in args}
    print('schedule: %s' % ' '.join(('STEP=%d' % s for (s, _, _) in steps)))
    print('args    : %s' % ','.join(args))
    worst = {cv: [0.0, 0.0, None] for cv in args}
    for r in rows:
        S = int(round(r[0] / a.dt))
        (eat, ekp) = at_step(steps, S)
        for (j, cv) in enumerate(args):
            (ci, ki) = cols[cv]
            dc = abs(r[ci] - eat[j])
            dk = abs(r[ki] - ekp[j])
            if dc > worst[cv][0]:
                worst[cv][0] = dc
                worst[cv][2] = S
            if dk > worst[cv][1]:
                worst[cv][1] = dk
    bad = []
    print('%-10s %14s %14s %12s' % ('CV', 'max|dcentre|', 'max|dkappa|', 'at step'))
    for cv in args:
        (dc, dk, S) = worst[cv]
        flag = '' if dc <= a.cntr_tol and dk <= a.kappa_tol else '  <-- FAIL'
        if flag:
            bad.append(cv)
        print('%-10s %14.3e %14.3e %12s%s' % (cv, dc, dk, S, flag))
    seg = d / 'PROGRESS.json'
    if seg.exists():
        p = json.loads(seg.read_text())
        print('segments: %s' % ', '.join(('%s[%s] %d-%d' % (x['segment'], x.get('status', 'ok'), x['start_step'], x['end_step']) for x in p.get('segments', []))))
    (t0, t1) = (rows[0][0], rows[-1][0])
    gaps = [(rows[i - 1][0], rows[i][0]) for i in range(1, len(rows)) if rows[i][0] - rows[i - 1][0] > 1.5 * (rows[1][0] - rows[0][0])]
    print('COLVAR  : %d rows, %.3f -> %.3f ps, %d gap(s)' % (len(rows), t0, t1, len(gaps)))
    for g in gaps[:5]:
        print('   gap %.3f -> %.3f ps' % g)
    if bad or gaps:
        print('RESUME VERIFICATION FAILED: %s%s' % ('centre/kappa mismatch in ' + ','.join(bad) if bad else '', '; %d COLVAR gap(s)' % len(gaps) if gaps else ''))
        return 1
    print('RESUME VERIFICATION PASS: centre and kappa track the original schedule at every reported step, and COLVAR is contiguous')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
