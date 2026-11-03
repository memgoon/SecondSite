# SPDX-License-Identifier: MIT
import argparse, json, os, sys, datetime, subprocess
from pathlib import Path
STRIP = ':WAT,:SOL,:HOH,:NA,:CL,:Na+,:Cl-,:CLA,:POT,:SOD'
LIG_MASK = ':296&!@/H'
PROT_MASK = '(:1-295)&!@/H&!:WAT,Na+,Cl-'
LIG_ALL = ':296'
PROT_ALL = ':1-295'
ANCHOR = ':1-295'
FRJ_SITE = {189, 190, 193, 194, 197, 201, 277, 278, 280, 281}
NTWX = 2500
DT_PS = 0.002
CONTACT_A = 4.5
DELTA_TOL = 0.001
CELL_TOL = 0.01
CLASH_A = 1.2
MINDIST_TOL = 0.05

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

def qc_frame(Xr, Xi, M, L, P, LA, PA, rid, h3x):
    import numpy as np
    flags = []
    rec = {}
    Minv = np.linalg.inv(M)
    deltas = {}
    for (nm, sl) in (('protein', PA), ('ligand', LA)):
        dl = Xi[sl] - Xr[sl]
        spread = float(np.abs(dl - dl.mean(axis=0)).max())
        rec[nm + '_delta_spread_A'] = spread
        if spread > DELTA_TOL:
            flags.append('%s was DEFORMED or split by imaging (per-atom translation varies by %.4f A)' % (nm, spread))
        deltas[nm] = dl.mean(axis=0)
        rec[nm + '_cells'] = [round(float(x), 4) for x in deltas[nm] @ Minv]
    rel = (deltas['ligand'] - deltas['protein']) @ Minv
    rec['relative_cells'] = [round(float(x), 4) for x in rel]
    rec['recentring_A'] = float(np.linalg.norm(deltas['protein']))
    if float(np.abs(rel - np.round(rel)).max()) > CELL_TOL:
        flags.append('the ligand moved relative to the protein by a NON-INTEGER number of cells (%s): that is not an imaging operation' % rec['relative_cells'])
    Dm = h3x.min_image_matrix(Xr[L], Xr[P], M)
    Dp = np.sqrt(((Xi[L][:, None, :] - Xi[P][None, :, :]) ** 2).sum(-1))
    mset = set(rid[np.where(Dm <= CONTACT_A)[1]].tolist())
    pset = set(rid[np.where(Dp <= CONTACT_A)[1]].tolist())
    rec.update(orig_min_image_pairs=int((Dm <= CONTACT_A).sum()), orig_min_image_min_A=float(Dm.min()), orig_frj=len(mset & FRJ_SITE), export_pairs=int((Dp <= CONTACT_A).sum()), export_min_A=float(Dp.min()), export_frj=len(pset & FRJ_SITE), residues_only_in_original=sorted(mset - pset), residues_only_in_export=sorted(pset - mset))
    if mset != pset:
        flags.append("the exported complex does not reproduce the frame's minimum-image contacts: %d residue(s) lost, %d gained" % (len(mset - pset), len(pset - mset)))
    if abs(rec['orig_min_image_min_A'] - rec['export_min_A']) > MINDIST_TOL:
        flags.append('minimum distance differs: original %.2f A vs export %.2f A' % (rec['orig_min_image_min_A'], rec['export_min_A']))
    d_raw = Xr[LA].mean(axis=0) - Xr[PA].mean(axis=0)
    d_exp = Xi[LA].mean(axis=0) - Xi[PA].mean(axis=0)
    d_near = d_raw - np.round(d_raw @ Minv) @ M
    rec['export_vs_nearest_copy_A'] = float(np.linalg.norm(d_exp - d_near))
    if rec['export_vs_nearest_copy_A'] > 1.0:
        flags.append('anchoring placed the ligand beside a periodic copy that is NOT the nearest one by centroid (relative displacement differs by %.2f A). autoimage anchors on the centre of the protein, which for a distant ligand need not be the nearest atom pair.' % rec['export_vs_nearest_copy_A'])
    allsel = np.concatenate([PA, LA])
    span = float(np.abs(Xi[allsel].max(axis=0) - Xi[allsel].min(axis=0)).max())
    rec['export_bounding_span_A'] = span
    perp = h3x.perp_widths(M)
    rec['box_min_perp_width_A'] = float(perp.min())
    if span > float(perp.min()):
        flags.append('the exported complex spans %.1f A, wider than the smallest periodic width %.1f A: it cannot be a single compact image' % (span, float(perp.min())))
    nb = float(np.sqrt(((Xi[LA][:, None, :] - Xi[PA][None, :, :]) ** 2).sum(-1)).min())
    rec['export_closest_pair_A'] = nb
    if nb < CLASH_A:
        flags.append('clash in the export: closest ligand-protein atom pair is %.2f A' % nb)
    rec['state'] = 'GENUINELY_DETACHED' if rec['orig_min_image_pairs'] == 0 else 'NEEDS_REVIEW' if flags else 'OK'
    return (rec, flags)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='<W>/<state>/production')
    ap.add_argument('--out', required=True, help='where the H5 coordinates go (NOT --dir)')
    ap.add_argument('--scratch', required=True, help='cpptraj workspace (NOT --dir)')
    ap.add_argument('--seeds-json', default=None, help='h4c SEEDS.json: chosen[].absolute_step')
    ap.add_argument('--steps', default=None, help='comma-separated absolute steps instead')
    ap.add_argument('--cpptraj', default=os.environ.get('CPPTRAJ_BIN') or 'cpptraj')
    ap.add_argument('--json', default=None, help='write the QC record here (outside --dir)')
    ap.add_argument('--no-pdb', action='store_true', help='QC only; write no PDBs')
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    for (name, p) in (('--out', Path(a.out).resolve()), ('--scratch', Path(a.scratch).resolve())):
        if p == d or str(p).startswith(str(d) + os.sep):
            sys.exit('REFUSING: %s is inside the production directory.' % name)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'raw').mkdir(exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    scr = Path(a.scratch).resolve() / ('run_' + stamp)
    scr.mkdir(parents=True, exist_ok=False)
    h3x = load_h3x(Path(__file__).resolve().parent)
    if h3x is None:
        sys.exit('h3x_ligand_escape.py must sit next to this script: the minimum-image and cell arithmetic is imported from it rather than duplicated.')
    pj = d / 'PROGRESS.json'
    if not pj.exists():
        sys.exit('no PROGRESS.json in %s' % d)
    segs = sorted(json.loads(pj.read_text()).get('segments', []), key=lambda x: x['segment'])
    if a.seeds_json:
        sj = json.loads(Path(a.seeds_json).read_text())
        want = [(c.get('snapshot'), int(c['absolute_step'])) for c in sj.get('chosen', [])]
        if not want:
            sys.exit('%s has no chosen[].absolute_step' % a.seeds_json)
    elif a.steps:
        want = [(i + 1, int(x)) for (i, x) in enumerate(a.steps.replace(',', ' ').split())]
    else:
        sys.exit('pass --seeds-json or --steps')
    print('seeds requested: %d   scratch %s' % (len(want), scr))
    import numpy as np, pytraj as pt
    nrow = []
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
        print('  seg_%s: %d seed frame(s)' % (tag, len(sel)))
        trajin = ['trajin %s %d %d 1' % (d / ('seg_%s.nc' % tag), f, f) for (_, _, f) in sel]
        raw = scr / ('raw_%s.nc' % tag)
        img = scr / ('img_%s.nc' % tag)
        dry = scr / 'dry.prmtop'
        if not dry.exists():
            (scr / 'dry.in').write_text('parm %s\nparmstrip %s\nparmwrite out %s\ngo\nquit\n' % (d / 'system.prmtop', STRIP, dry))
            run_cpptraj(a.cpptraj, scr / 'dry.in', scr, [dry], 'stripped topology')
        (scr / ('raw_%s.in' % tag)).write_text('\n'.join(['parm %s' % (d / 'system.prmtop')] + trajin + ['strip %s' % STRIP, 'trajout %s netcdf' % raw, 'go', 'quit', '']))
        run_cpptraj(a.cpptraj, scr / ('raw_%s.in' % tag), scr, [raw], 'raw export seg_%s' % tag)
        (scr / ('img_%s.in' % tag)).write_text('\n'.join(['parm %s' % (d / 'system.prmtop')] + trajin + ['autoimage anchor %s' % ANCHOR, 'strip %s' % STRIP, 'trajout %s netcdf' % img, 'go', 'quit', '']))
        run_cpptraj(a.cpptraj, scr / ('img_%s.in' % tag), scr, [img], 'imaged export seg_%s' % tag)
        tr = pt.iterload(str(raw), top=str(dry))
        ti = pt.iterload(str(img), top=str(dry))
        if tr.n_frames != ti.n_frames or tr.n_frames != len(sel):
            sys.exit('frame count mismatch for seg_%s: raw %d, imaged %d, requested %d' % (tag, tr.n_frames, ti.n_frames, len(sel)))
        tp = tr.top
        L = tp.select(LIG_MASK)
        P = tp.select(PROT_MASK)
        LA = tp.select(LIG_ALL)
        PA = tp.select(PROT_ALL)
        if len(L) == 0 or len(P) == 0:
            sys.exit('ligand or protein mask selected nothing')
        rid = np.array([tp.atom(i).resid + 1 for i in P])
        for (k, ((label, st_, fr), fa, fb)) in enumerate(zip(sel, tr, ti)):
            box = np.asarray(fa.box.values if hasattr(fa.box, 'values') else fa.box, dtype=float)
            (rec, flags) = qc_frame(fa.xyz, fb.xyz, h3x.cell_vectors(box), L, P, LA, PA, rid, h3x)
            rec.update(seed=label, step=st_, ns=round(st_ * DT_PS / 1000.0, 6), segment=tag, frame=fr, flags=flags)
            nrow.append(rec)
            if not a.no_pdb:
                for (src, dst) in ((img, out / ('snapshot_%s.pdb' % label)), (raw, out / 'raw' / ('snapshot_%s.pdb' % label))):
                    pi = scr / ('w_%s_%s.in' % (tag, label))
                    pi.write_text('parm %s\ntrajin %s %d %d 1\ntrajout %s pdb\ngo\nquit\n' % (dry, src, k + 1, k + 1, dst))
                    run_cpptraj(a.cpptraj, pi, scr, [dst], 'pdb for seed %s' % label)
    if not nrow:
        sys.exit('none of the requested steps fell inside a valid segment')
    nrow.sort(key=lambda r: r['step'])
    byst = {}
    for r_ in nrow:
        byst.setdefault(r_['state'], []).append(r_)
    print('\n%-6s %10s %9s %8s %5s %8s %5s %9s %-14s %s' % ('seed', 'ns', 'origPairs', 'origMin', 'FRJ', 'expPairs', 'FRJ', 'copyDiff', 'relCells', 'state'))
    for r_ in nrow:
        print('%-6s %10.3f %9d %8.2f %5d %8d %5d %9.2f %-14s %s' % (r_['seed'], r_['ns'], r_['orig_min_image_pairs'], r_['orig_min_image_min_A'], r_['orig_frj'], r_['export_pairs'], r_['export_frj'], r_['export_vs_nearest_copy_A'], str(r_.get('relative_cells')), r_['state']))
        for f in r_['flags']:
            print('       ! %s' % f)
    print('\nsummary over %d seed frame(s):' % len(nrow))
    for k in ('OK', 'GENUINELY_DETACHED', 'NEEDS_REVIEW'):
        if k in byst:
            print('  %-20s %3d' % (k, len(byst[k])))
    miss = len(want) - len(nrow)
    if miss:
        print('  %-20s %3d  (requested steps outside every valid segment)' % ('NOT_EXPORTED', miss))
    print('\nThis is a coordinate REPRESENTATION correction for resolvation, recorded as such.')
    print("It is not the authors' helper verbatim: theirs, and h4c, strip only.")
    print('GENUINELY_DETACHED frames were exported as the unbound states they are; nothing')
    print('was re-docked or nudged. Whether such seeds may be used is decided elsewhere.')
    if a.json:
        Path(a.json).write_text(json.dumps(dict(dir=str(d), out=str(out), scratch=str(scr), cutoff_A=CONTACT_A, delta_tol_A=DELTA_TOL, cell_tol_cells=CELL_TOL, requested=len(want), exported=len(nrow), counts={k: len(v) for (k, v) in byst.items()}, frames=nrow), indent=1, ensure_ascii=False) + '\n')
        print('\nrecord -> %s' % a.json)
    return 0 if not byst.get('NEEDS_REVIEW') else 1
if __name__ == '__main__':
    sys.exit(main())
