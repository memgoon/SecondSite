# SPDX-License-Identifier: MIT
import argparse, json, math, os, re, sys
from pathlib import Path
STRIP = ':WAT,:SOL,:HOH,:NA,:CL,:Na+,:Cl-,:CLA,:POT,:SOD'

def fail(msg):
    print('  FAIL: ' + msg)
    sys.exit(1)

def ok(msg):
    print('  ok   ' + msg)

def nc_header(path):
    from parmed.amber import NetCDFTraj
    n = NetCDFTraj.open_old(str(path))
    return n

def kept_indices(wet_prmtop):
    import parmed as pmd
    from parmed.amber import AmberMask
    p = pmd.load_file(str(wet_prmtop))
    sel = AmberMask(p, STRIP).Selection()
    return [i for (i, v) in enumerate(sel) if not v]

def bond_signature(parm):
    out = []
    for b in parm.bonds:
        (i, j) = sorted((b.atom1.idx, b.atom2.idx))
        k = round(b.type.k, 6) if b.type is not None else None
        r = round(b.type.req, 6) if b.type is not None else None
        out.append((i, j, k, r))
    return sorted(out)

def stripped_atom_list(wet_prmtop):
    import parmed as pmd
    p = pmd.load_file(str(wet_prmtop))
    p.strip(STRIP)
    return [(a.name, a.residue.name, a.residue.idx, round(a.charge, 6), a.type) for a in p.atoms]

def dry_atom_list(dry_prmtop):
    import parmed as pmd
    p = pmd.load_file(str(dry_prmtop))
    return [(a.name, a.residue.name, a.residue.idx, round(a.charge, 6), a.type) for a in p.atoms]

def verify_md(d, nstlim, ntwx):
    out = d / 'production.out'
    rst = d / 'production.rst'
    nc = d / 'production.nc'
    for f in (out, rst, nc):
        if not f.exists() or f.stat().st_size == 0:
            fail('missing or empty %s' % f.name)
    txt = out.read_text(errors='replace')
    for tok in ('NaN', 'vlimit exceeded'):
        if tok in txt:
            fail('%s in production.out' % tok)
    hits = re.findall('NSTEP =\\s*(\\d+)', txt)
    if not hits:
        fail('no NSTEP lines in production.out')
    if int(hits[-1]) != nstlim:
        fail('Amber reports NSTEP=%s, requested %d' % (hits[-1], nstlim))
    ok('production.out: NSTEP %d, no NaN or vlimit' % nstlim)
    exp_frames = nstlim // ntwx
    h = nc_header(nc)
    if h.frame != exp_frames:
        fail('production.nc header says %d frames, expected %d' % (h.frame, exp_frames))
    import parmed as pmd
    nwet = len(pmd.load_file(str(d / 'system.prmtop')).atoms)
    if h.atom != nwet:
        fail('production.nc header says %d atoms, system.prmtop has %d' % (h.atom, nwet))
    ok('production.nc header: %d frames x %d atoms (independent of any topology)' % (h.frame, h.atom))
    t = open(rst).read().splitlines()
    if len(t) < 2:
        fail('production.rst is truncated')
    parts = t[1].split()
    if len(parts) >= 2:
        ps = float(parts[1])
        expect = nstlim * 0.002
        if abs(ps - expect) > 2 * 0.002:
            fail('production.rst records %.3f ps, expected %.3f' % (ps, expect))
        ok('production.rst records %.3f ps' % ps)
    return exp_frames

def verify_dry(d, nstlim, ntwx, sample=25, coord_tol=0.002):
    ev = {'nstlim': nstlim, 'ntwx': ntwx, 'coord_tol_A': coord_tol, 'topology_equal': False, 'frames': None, 'atoms': None, 'frames_sampled': 0, 'wet_present': False, 'wet_dry_max_dev_A': None, 'wet_dry_coords_verified': False, 'time_axis': None}

    def dump():
        (d / 'h5v_dry.json').write_text(json.dumps(ev, indent=2) + '\n')
    import numpy as np
    exp_frames = nstlim // ntwx
    dry = d / 'production_dry.nc'
    if not dry.exists() or dry.stat().st_size == 0:
        fail('missing or empty production_dry.nc')
    want = stripped_atom_list(d / 'system.prmtop')
    got = dry_atom_list(d / 'system_dry.prmtop')
    if len(want) != len(got):
        fail('system_dry.prmtop has %d atoms but stripping system.prmtop gives %d' % (len(got), len(want)))
    bad = [i for (i, (x, y)) in enumerate(zip(want, got)) if x != y]
    if bad:
        i = bad[0]
        fail('system_dry.prmtop atom %d is %s but stripping system.prmtop gives %s (%d differ)' % (i, got[i], want[i], len(bad)))
    ev['topology_equal'] = True
    ev['atoms'] = len(got)
    dump()
    ok('system_dry.prmtop == system.prmtop stripped of %s: %d atoms, names/residues/types/charges all match' % (STRIP, len(got)))
    h = nc_header(dry)
    if h.atom != len(want):
        fail('production_dry.nc header says %d atoms, the stripped topology has %d' % (h.atom, len(want)))
    if h.frame != exp_frames:
        fail('production_dry.nc header says %d frames, expected %d' % (h.frame, exp_frames))
    ev['frames'] = h.frame
    dump()
    ok('production_dry.nc header: %d frames x %d atoms' % (h.frame, h.atom))
    idx = sorted(set((int(round(x)) for x in np.linspace(0, exp_frames - 1, min(sample, exp_frames)))))
    import pytraj as pt
    t = pt.iterload(str(dry), top=str(d / 'system_dry.prmtop'))
    checked = 0
    for i in idx:
        xyz = t[i].xyz
        if xyz.shape[0] != len(want):
            fail('frame %d holds %d atoms, expected %d' % (i, xyz.shape[0], len(want)))
        if not np.isfinite(xyz).all():
            fail('frame %d has non-finite coordinates' % i)
        if float(np.abs(xyz).max()) == 0.0:
            fail('frame %d is all zeros' % i)
        checked += 1
    ev['frames_sampled'] = checked
    dump()
    ok('%d frames sampled across the file: finite, right shape, non-zero' % checked)
    wet = d / 'production.nc'
    if wet.exists() and wet.stat().st_size > 0:
        keep = kept_indices(d / 'system.prmtop')
        if len(keep) != len(want):
            fail('the strip mask keeps %d atoms but the stripped topology has %d' % (len(keep), len(want)))
        wt = pt.iterload(str(wet), top=str(d / 'system.prmtop'))
        if wt.n_frames != exp_frames:
            fail('production.nc holds %d frames, expected %d' % (wt.n_frames, exp_frames))
        worst = 0.0
        worst_at = None
        for i in idx:
            a = np.asarray(wt[i].xyz)[keep]
            b = np.asarray(t[i].xyz)
            if a.shape != b.shape:
                fail('frame %d: wet keeps %s atoms, dry has %s' % (i, a.shape, b.shape))
            dmax = float(np.abs(a - b).max())
            if dmax > worst:
                worst = dmax
                worst_at = i
        if worst > coord_tol:
            fail('wet and dry coordinates disagree by %.4g A at frame %s (tolerance %.4g). The dry copy is NOT the wet trajectory stripped.' % (worst, worst_at, coord_tol))
        ev['wet_present'] = True
        ev['wet_dry_max_dev_A'] = worst
        ev['wet_dry_coords_verified'] = True
        dump()
        ok('wet -> dry coordinates agree atom for atom over %d frames, worst %.2e A (tolerance %.0e)' % (len(idx), worst, coord_tol))
    else:
        dump()
        print('  NOTE production.nc is gone, so the wet->dry coordinate check could NOT run and this dry file cannot be proven faithful here. Completion must rely on the evidence recorded when the wet file still existed.')
    tm = None
    try:
        tm = np.asarray(h.time) if h.time is not None else None
    except Exception:
        tm = None
    if tm is not None and len(tm) == exp_frames:
        dts = np.diff(tm)
        step = ntwx * 0.002
        if len(dts) and (not np.isfinite(dts).all() or abs(float(dts.mean()) - step) > 0.001 or float(dts.min()) <= 0):
            fail('time axis is not monotonic at %g ps (mean step %.4f)' % (step, float(dts.mean())))
        ev['time_axis'] = 'monotonic, %g ps' % step
        dump()
        ok('time axis: %d points, monotonic, %g ps apart' % (len(tm), step))
    else:
        print('  note the dry file carries no usable time axis; frame count and coordinates were checked instead')
    return exp_frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('md', 'dry'), required=True)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--nstlim', type=int, required=True)
    ap.add_argument('--ntwx', type=int, required=True)
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    print('h5v %s  %s' % (a.mode, d))
    n = verify_md(d, a.nstlim, a.ntwx) if a.mode == 'md' else verify_dry(d, a.nstlim, a.ntwx)
    print('  => PASS (%d frames)' % n)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
