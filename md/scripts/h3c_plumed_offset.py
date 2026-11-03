# SPDX-License-Identifier: MIT
import argparse, re, sys

def parse(lines):
    steps = []
    for (i, l) in enumerate(lines):
        m = re.match('\\s*STEP(\\d+)\\s*=\\s*(\\d+)\\s+AT\\1\\s*=\\s*([^\\s]+)\\s+KAPPA\\1\\s*=\\s*([^\\s]+)', l)
        if m:
            steps.append(dict(idx=int(m.group(1)), line=i, step=int(m.group(2)), at=[float(x) for x in m.group(3).split(',')], kappa=[float(x) for x in m.group(4).split(',')]))
    return steps

def interp(steps, S):
    if S <= steps[0]['step']:
        return (steps[0]['at'], steps[0]['kappa'])
    if S >= steps[-1]['step']:
        return (steps[-1]['at'], steps[-1]['kappa'])
    for (a, b) in zip(steps, steps[1:]):
        if a['step'] <= S <= b['step']:
            f = (S - a['step']) / (b['step'] - a['step'])
            at = []
            for (x, y) in zip(a['at'], b['at']):
                d = y - x
                at.append(x + f * d)
            kp = [x + f * (y - x) for (x, y) in zip(a['kappa'], b['kappa'])]
            return (at, kp)
    raise ValueError('step %d outside the schedule' % S)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plumed', required=True)
    ap.add_argument('--offset', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--colvar', default=None, help='rewrite the PRINT line to this FILE= (one COLVAR per segment)')
    ap.add_argument('--restart', action='store_true', help='prepend RESTART; not needed when each leg has its own COLVAR')
    a = ap.parse_args()
    lines = open(a.plumed).read().splitlines()
    steps = parse(lines)
    if not steps:
        sys.exit('no STEPn/ATn/KAPPAn lines found in ' + a.plumed)
    (at0, kp0) = interp(steps, a.offset)
    fmt = lambda v: ','.join(('%.6f' % x for x in v))
    out = []
    n = 0
    for (i, l) in enumerate(lines):
        s = next((x for x in steps if x['line'] == i), None)
        if s is None:
            out.append(l)
            continue
        if n == 0:
            out.append('  STEP0=0    AT0=%s    KAPPA0=%s' % (fmt(at0), fmt(kp0)))
            n = 1
        rel = s['step'] - a.offset
        if rel <= 0:
            continue
        out.append('  STEP%d=%d    AT%d=%s    KAPPA%d=%s' % (n, rel, n, fmt(s['at']), n, fmt(s['kappa'])))
        n += 1
    if a.colvar:
        out = [re.sub('FILE=\\S+', 'FILE=%s' % a.colvar, l) if l.strip().startswith('PRINT') else l for l in out]
        if not any((l.strip().startswith('PRINT') for l in out)):
            raise SystemExit('no PRINT line to retarget in ' + a.plumed)
    if a.restart:
        out = ['RESTART'] + out
    open(a.out, 'w').write('\n'.join(out) + '\n')
    print('offset %d -> %s  (anchor AT0=%s KAPPA0=%s, %d schedule points%s)' % (a.offset, a.out, fmt(at0), fmt(kp0), n, ', COLVAR=%s' % a.colvar if a.colvar else ''))
if __name__ == '__main__':
    main()
