# SPDX-License-Identifier: MIT
import argparse, json, os, sys, datetime, subprocess
from pathlib import Path
STRIP = ':WAT,:SOL,:HOH,:NA,:CL,:Na+,:Cl-,:CLA,:POT,:SOD'
LIG_MASK = ':296&!@/H'
PROT_MASK = '(:1-295)&!@/H&!:WAT,Na+,Cl-'
LIG_ALL = ':296'
PROT_ALL = ':1-295'
FIT_MASK = ':1-282&(!@/H)&(!:198)'
ANCHOR = ':1-295'
FRJ_SITE = {189, 190, 193, 194, 197, 201, 277, 278, 280, 281}
NTWX = 2500
DT_PS = 0.002
CONTACT_A = 4.5
CLASH_BINS = (2.0, 2.5, 3.0)

def sh(cmd, cwd):
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True, text=True)

def load_h3x(here):
    import importlib.util
    for c in (here / 'h3x_ligand_escape.py', here.parent / 'ops' / 'h3x_ligand_escape.py'):
        if c.exists():
            sp = importlib.util.spec_from_file_location('h3xmod', c)
            m = importlib.util.module_from_spec(sp)
            argv = sys.argv
            sys.argv = [argv[0]]
            sp.loader.exec_module(m)
            sys.argv = argv
            return m
    return None

def run_cpptraj(binary, inp, cwd, outs, what):
    for o in outs:
        if Path(o).exists():
            sys.exit('REFUSING: %s exists before %s ran; the scratch must be fresh.' % (o, what))
    r = sh([binary, '-i', inp], cwd=cwd)
    if r.returncode != 0:
        sys.exit('cpptraj FAILED (%s, status %d):\n%s' % (what, r.returncode, r.stderr[-2000:]))
    for o in outs:
        if not Path(o).exists() or Path(o).stat().st_size == 0:
            sys.exit('cpptraj reported success but %s is missing or empty (%s)' % (o, what))
    return r

def kabsch(P, Q):
    import numpy as np
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    (U, S, Vt) = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).T
    return (R, qc - pc @ R)

def rmsd(A, B):
    import numpy as np
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))

def topo_signature(top, sel):
    return [(top.atom(i).name, top.atom(i).resid + 1, top.residue(top.atom(i).resid).name) for i in sel]

def quantiles(v):
    import numpy as np
    if not v:
        return {}
    a = np.asarray(v, dtype=float)
    q = lambda p: float(np.percentile(a, p))
    return dict(n=len(v), min=float(a.min()), p10=q(10), p25=q(25), median=q(50), p75=q(75), p90=q(90), max=float(a.max()), mean=float(a.mean()))

def pose_metrics(Xi, refX, F, L, LA, P, rid, ref_contacts):
    import numpy as np
    (R, t) = kabsch(Xi[F], refX[F])
    Y = Xi @ R + t
    D = np.sqrt(((Y[L][:, None, :] - Y[P][None, :, :]) ** 2).sum(-1))
    cset = set(rid[np.where(D <= CONTACT_A)[1]].tolist())
    kept = len(cset & ref_contacts) / len(ref_contacts) if ref_contacts else float('nan')
    rfrj = ref_contacts & FRJ_SITE
    (Rl, tl) = kabsch(Xi[L], refX[L])
    return dict(core_fit_rmsd_A=rmsd(Y[F], refX[F]), lig_rmsd_A=rmsd(Y[L], refX[L]), centroid_A=float(np.linalg.norm(Y[LA].mean(axis=0) - refX[LA].mean(axis=0))), kept_contacts=kept, kept_frj=len(cset & rfrj) / len(rfrj) if rfrj else float('nan'), min_pair_A=float(D.min()), internal_rmsd_A=rmsd(Xi[L] @ Rl + tl, refX[L]), pairs_below={'%.1f' % b: int((D < b).sum()) for b in CLASH_BINS})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='<W>/<state>/production')
    ap.add_argument('--scratch', required=True, help='cpptraj workspace (NOT --dir)')
    ap.add_argument('--seeds-json', default=None)
    ap.add_argument('--steps', default=None)
    ap.add_argument('--ref-docked-parm', default=None)
    ap.add_argument('--ref-docked-crd', default=None)
    ap.add_argument('--ref-equil-parm', default=None)
    ap.add_argument('--ref-equil-crd', default=None)
    ap.add_argument('--fit-mask', default=FIT_MASK)
    ap.add_argument('--cpptraj', default=os.environ.get('CPPTRAJ_BIN') or 'cpptraj')
    ap.add_argument('--json', default=None)
    ap.add_argument('--print-frames', action='store_true', help='also print every frame')
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    scr0 = Path(a.scratch).resolve()
    if scr0 == d or str(scr0).startswith(str(d) + os.sep):
        sys.exit('REFUSING: --scratch is inside the production directory.')
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    scr = scr0 / ('run_' + stamp)
    scr.mkdir(parents=True, exist_ok=False)
    h3x = load_h3x(Path(__file__).resolve().parent)
    if h3x is None:
        sys.exit('h3x_ligand_escape.py must sit next to this script (cell arithmetic).')
    refs = []
    for (nm, pk, ck) in (('docked', 'ref_docked_parm', 'ref_docked_crd'), ('equilibrated', 'ref_equil_parm', 'ref_equil_crd')):
        pp = getattr(a, pk)
        cc = getattr(a, ck)
        if pp and cc:
            refs.append((nm, Path(pp).resolve(), Path(cc).resolve()))
        elif pp or cc:
            sys.exit('--%s and --%s must be given together' % (pk.replace('_', '-'), ck.replace('_', '-')))
    if not refs:
        sys.exit('give at least one reference. Both are expected: the pre-minimisation docked complex AND the equilibrated starting structure - the review asks for the pose against both, and they are different structures.')
    pj = d / 'PROGRESS.json'
    if not pj.exists():
        sys.exit('no PROGRESS.json in %s' % d)
    _prog = json.loads(pj.read_text())
    state = _prog.get('state')
    segs = sorted(_prog.get('segments', []), key=lambda x: x['segment'])
    if a.seeds_json:
        sj = json.loads(Path(a.seeds_json).read_text())
        want = [(c.get('snapshot'), int(c['absolute_step'])) for c in sj.get('chosen', [])]
    elif a.steps:
        want = [(i + 1, int(x)) for (i, x) in enumerate(a.steps.replace(',', ' ').split())]
    else:
        sys.exit('pass --seeds-json or --steps')
    if not want:
        sys.exit('no steps requested')
    print('pose report   %d seed frame(s)   references: %s' % (len(want), ', '.join((n for (n, _, _) in refs))))
    print('NOTE: distributions only. No pass threshold is applied or implied anywhere in this output.')
    import numpy as np, pytraj as pt
    dry = scr / 'dry.prmtop'
    (scr / 'dry.in').write_text('parm %s\nparmstrip %s\nparmwrite out %s\ngo\nquit\n' % (d / 'system.prmtop', STRIP, dry))
    run_cpptraj(a.cpptraj, scr / 'dry.in', scr, [dry], 'stripped topology')
    tref = pt.load_topology(str(dry))
    L = tref.select(LIG_MASK)
    P = tref.select(PROT_MASK)
    LA = tref.select(LIG_ALL)
    PA = tref.select(PROT_ALL)
    F = tref.select(a.fit_mask)
    for (nm, sel) in (('ligand heavy', L), ('protein heavy', P), ('fit mask', F)):
        if len(sel) == 0:
            sys.exit('%s selected nothing on the stripped topology' % nm)
    print('  stripped topology: %d ligand heavy, %d protein heavy, %d fit atoms' % (len(L), len(P), len(F)))
    rid = np.array([tref.atom(i).resid + 1 for i in P])
    sig = topo_signature(tref, range(tref.n_atoms))
    REF = {}
    for (nm, pp, cc) in refs:
        try:
            rt = pt.load(str(cc), top=str(pp))
        except Exception as e:
            sys.exit('could not load the %s reference: %s' % (nm, e))
        rs = pt.strip(rt, STRIP) if rt.top.n_atoms != tref.n_atoms else rt
        if rs.top.n_atoms != tref.n_atoms:
            sys.exit('%s reference has %d atoms after stripping, the trajectory topology has %d. Atom correspondence cannot be established; refusing rather than matching by name.' % (nm, rs.top.n_atoms, tref.n_atoms))
        rsig = topo_signature(rs.top, range(rs.top.n_atoms))
        if rsig != sig:
            diff = [i for (i, (x, y)) in enumerate(zip(rsig, sig)) if x != y]
            sys.exit('%s reference does not correspond atom for atom: %d difference(s), first at index %d (%s vs %s). Refusing; nothing is reordered.' % (nm, len(diff), diff[0], rsig[diff[0]], sig[diff[0]]))
        X = rs.xyz[0]
        Dr = np.sqrt(((X[L][:, None, :] - X[P][None, :, :]) ** 2).sum(-1))
        cset = set(rid[np.where(Dr <= CONTACT_A)[1]].tolist())
        REF[nm] = dict(X=X, contacts=cset, frj=len(cset & FRJ_SITE), min_pair=float(Dr.min()), parm=str(pp), crd=str(cc))
        print('  %-13s %d atoms verified atom-for-atom; %d contacted residue(s), %d in the FRJ site, closest pair %.2f A' % (nm, tref.n_atoms, len(cset), len(cset & FRJ_SITE), float(Dr.min())))
    rows = []
    for s in segs:
        tag = s['segment']
        start = int(s['start_step'])
        lim = s.get('valid_to_step') or s.get('end_step')
        if lim is None:
            continue
        lim = int(lim)
        nframe = (lim - start) // NTWX
        sel = []
        for (label, st_) in want:
            off = st_ - start
            if off <= 0 or off % NTWX:
                continue
            fr = off // NTWX
            if 1 <= fr <= nframe:
                sel.append((label, st_, fr))
        if not sel:
            continue
        trajin = ['trajin %s %d %d 1' % (d / ('seg_%s.nc' % tag), f, f) for (_, _, f) in sel]
        raw = scr / ('raw_%s.nc' % tag)
        img = scr / ('img_%s.nc' % tag)
        (scr / ('raw_%s.in' % tag)).write_text('\n'.join(['parm %s' % (d / 'system.prmtop')] + trajin + ['strip %s' % STRIP, 'trajout %s netcdf' % raw, 'go', 'quit', '']))
        run_cpptraj(a.cpptraj, scr / ('raw_%s.in' % tag), scr, [raw], 'raw seg_%s' % tag)
        (scr / ('img_%s.in' % tag)).write_text('\n'.join(['parm %s' % (d / 'system.prmtop')] + trajin + ['autoimage anchor %s' % ANCHOR, 'strip %s' % STRIP, 'trajout %s netcdf' % img, 'go', 'quit', '']))
        run_cpptraj(a.cpptraj, scr / ('img_%s.in' % tag), scr, [img], 'imaged seg_%s' % tag)
        tr = pt.iterload(str(raw), top=str(dry))
        ti = pt.iterload(str(img), top=str(dry))
        if not tr.n_frames == ti.n_frames == len(sel):
            sys.exit('frame count mismatch in seg_%s' % tag)
        for ((label, st_, fr), fa, fb) in zip(sel, tr, ti):
            box = np.asarray(fa.box.values if hasattr(fa.box, 'values') else fa.box, dtype=float)
            M = h3x.cell_vectors(box)
            Xr = fa.xyz
            Xi = fb.xyz
            shift = (Xi[LA] - Xr[LA]).mean(axis=0) @ np.linalg.inv(M)
            rec = dict(seed=label, step=st_, ns=round(st_ * DT_PS / 1000.0, 6), segment=tag, frame=fr, imaging_cells=[round(float(x), 3) for x in shift], needed_imaging=bool(np.abs(np.round(shift)).max() > 0))
            for (nm, r) in REF.items():
                rec[nm] = pose_metrics(Xi, r['X'], F, L, LA, P, rid, r['contacts'])
            rows.append(rec)
    if not rows:
        sys.exit('none of the requested steps fell inside a valid segment')
    rows.sort(key=lambda r: r['step'])
    if a.print_frames:
        for nm in REF:
            print('\nper frame, vs %s:' % nm)
            print('  %-6s %10s %9s %9s %9s %9s %9s' % ('seed', 'ns', 'coreFit', 'ligRMSD', 'centroid', 'kept', 'minPair'))
            for r in rows:
                v = r[nm]
                print('  %-6s %10.3f %9.2f %9.2f %9.2f %9.2f %9.2f' % (r['seed'], r['ns'], v['core_fit_rmsd_A'], v['lig_rmsd_A'], v['centroid_A'], v['kept_contacts'], v['min_pair_A']))
    print('\nimaging: %d of %d frame(s) needed a whole-cell shift to be read as a complex' % (sum((1 for r in rows if r['needed_imaging'])), len(rows)))
    dist = {}
    for nm in REF:
        dist[nm] = {}
        print('\ndistribution vs %s  (n=%d)' % (nm, len(rows)))
        print('  %-18s %8s %8s %8s %8s %8s %8s %8s' % ('metric', 'min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'))
        for (key, label) in (('core_fit_rmsd_A', 'core fit RMSD A'), ('lig_rmsd_A', 'ligand RMSD A'), ('centroid_A', 'centroid shift A'), ('kept_contacts', 'kept contacts'), ('kept_frj', 'kept FRJ contacts'), ('min_pair_A', 'closest pair A'), ('internal_rmsd_A', 'ligand internal A')):
            q = quantiles([r[nm][key] for r in rows if r[nm][key] == r[nm][key]])
            dist[nm][key] = q
            if q:
                print('  %-18s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f' % (label, q['min'], q['p10'], q['p25'], q['median'], q['p75'], q['p90'], q['max']))
        print('  clash bins (descriptive, not criteria): ligand-protein heavy pairs below')
        for b in CLASH_BINS:
            v = [r[nm]['pairs_below']['%.1f' % b] for r in rows]
            nz = sum((1 for x in v if x > 0))
            print('    < %.1f A : %d frame(s) have any, worst frame has %d pair(s)' % (b, nz, max(v) if v else 0))
    print('\nNo verdict is given. Any threshold on these numbers must be written down beforehand, with its justification, somewhere other than this output.')
    if a.json:
        Path(a.json).write_text(json.dumps(dict(dir=str(d), state=state, scratch=str(scr), fit_mask=a.fit_mask, cutoff_A=CONTACT_A, clash_bins=list(CLASH_BINS), thresholds='NONE BY DESIGN', references={k: dict(parm=v['parm'], crd=v['crd'], contact_residues=sorted(v['contacts']), frj_contacts=v['frj'], min_pair_A=v['min_pair']) for (k, v) in REF.items()}, requested=len(want), reported=len(rows), distribution=dist, frames=rows), indent=1, ensure_ascii=False) + '\n')
        print('\nrecord -> %s' % a.json)
    return 0
if __name__ == '__main__':
    sys.exit(main())
