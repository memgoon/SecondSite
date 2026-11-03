# SPDX-License-Identifier: MIT
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
STRIP = ':WAT,:SOL,:HOH,:NA,:CL,:Na+,:Cl-,:CLA,:POT,:SOD'
LIG_MASK = ':296&!@/H'
PROT_MASK = '(:1-295)&!@/H&!:WAT,Na+,Cl-'
PROT_ALL = ':1-295'
LIG_ALL = ':296'
FIT_MASK = ':1-282&(!@/H)&(!:198)'
WPD_MASK = ':179-185@CA,C,N,O'
FRJ_SITE = {189, 190, 193, 194, 197, 201, 277, 278, 280, 281}
CONTACT_A = 4.5
REF_MD5 = '1d9ad04e386db0298f4b94a95446d9d6'
OUTS = ('system.prmtop', 'system_dry.prmtop', 'seed_equil.rst7')

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda : f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def sib(here, fname, mod):
    for c in (here / fname, here.parent / 'ops' / fname):
        if c.exists():
            sp = importlib.util.spec_from_file_location(mod, c)
            m = importlib.util.module_from_spec(sp)
            argv = sys.argv
            sys.argv = [argv[0]]
            sp.loader.exec_module(m)
            sys.argv = argv
            return m
    return None

def sig_of(path):
    sig = None
    outs = {}
    snap = None
    for ln in Path(path).read_text().splitlines():
        if ln.startswith('signature='):
            sig = ln.split('=', 1)[1]
        elif ln.startswith('out_'):
            (k, _, v) = ln[4:].partition('=')
            outs[k] = v
        elif ln.startswith('snapshot='):
            snap = ln.split('=', 1)[1]
    toks = {}
    if sig:
        for tok in sig.split():
            (k, _, v) = tok.partition('=')
            if k:
                toks[k] = v
    return (sig, outs, snap, toks.get('snap'), toks)

def expected_inputs(root, leg):
    r = Path(root)
    return {'mol2': r / 'prepared_gmx' / ('2e__%s' % leg) / 'ligand.mol2', 'frcmod': r / 'prepared_gmx' / ('2e__%s' % leg) / 'ligand.frcmod', 'parent': r / 'params' / 'parent_dry.prmtop', 'em': r / 'mdp' / 'em.mdp', 'heat': r / 'mdp' / 'heat.mdp', 'npt': r / 'mdp' / 'npt.mdp'}

def topo_sig(top, sel):
    return [(top.atom(i).name, top.residue(top.atom(i).resid).name, top.atom(i).resid) for i in sel]

def quant(v):
    import numpy as np
    if not v:
        return {}
    a = np.asarray(v, dtype=float)
    f = lambda p: float(np.percentile(a, p))
    return dict(n=len(v), min=float(a.min()), p10=f(10), median=f(50), p90=f(90), max=float(a.max()), mean=float(a.mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds-root', required=True)
    ap.add_argument('--manifest', default=None, help='a set_*.json from h5m_seed_sets')
    ap.add_argument('--plan', default=None, help='a dispatch PLAN.json (hosts[*] merged)')
    ap.add_argument('--host', default=None, help='with --plan, restrict to this host')
    ap.add_argument('--seeds-file', default=None, help="selector file 'leg:num,...'")
    ap.add_argument('--ref', required=True, help="the authors' closed.pdb")
    ap.add_argument('--strata', default=None, help='seed_strata.json, for the before values')
    ap.add_argument('--pose-ref-root', default=None, help='directory holding 2e__<leg>/ with dry.{prmtop,inpcrd} and system.prmtop + equil.rst7. Without it the pose section is reported as NOT EVALUATED rather than silently skipped.')
    ap.add_argument('--bond-max', type=float, default=3.0, help='longest acceptable bonded distance in A before the solute is called incoherent (default 3.0)')
    ap.add_argument('--inputs-root', default='.', help='package root holding prepared_gmx/, params/ and mdp/ for the signature recheck')
    ap.add_argument('--allow-unpinned-ref', action='store_true', help='proceed when --ref is not the pinned authors closed.pdb')
    ap.add_argument('--print-frames', action='store_true')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    h3x = sib(here, 'h3x_ligand_escape.py', 'h3xm')
    h4p = sib(here, 'h4p_pose_report.py', 'h4pm')
    if h3x is None or h4p is None:
        sys.exit('h3x_ligand_escape.py and h4p_pose_report.py must sit next to this script: the minimum-image, cell and superposition maths is imported, not copied.')
    given = [x for x in (a.manifest, a.plan, a.seeds_file) if x]
    if len(given) != 1:
        sys.exit('give exactly one of --manifest, --plan or --seeds-file (got %d)' % len(given))
    wpd_of = lambda r: next((r[k] for k in ('wpd_rmsd_to_closed_A', 'wpd') if r.get(k) is not None), None)
    if a.manifest:
        M = json.loads(Path(a.manifest).read_text())
        seeds = [dict(leg=r['leg'], seed=r['seed'], wpd_before=wpd_of(r)) for r in M.get('seeds', [])]
        label = M.get('name', a.manifest)
        cond = M.get('condition', '(none recorded)')
    elif a.plan:
        P = json.loads(Path(a.plan).read_text())
        hosts = P.get('hosts', {})
        if a.host:
            if a.host not in hosts:
                sys.exit('%s has no host %r' % (a.plan, a.host))
            hosts = {a.host: hosts[a.host]}
        seeds = [dict(leg=r['leg'], seed=r['seed'], wpd_before=wpd_of(r)) for h in sorted(hosts) for r in hosts[h]]
        label = '%s%s' % (a.plan, ' host=%s' % a.host if a.host else ' (all hosts)')
        cond = P.get('policy', '(no policy recorded)')
        if P.get('source_sha256'):
            print('plan source_sha256 %s' % P['source_sha256'])
    else:
        seeds = []
        for tok in Path(a.seeds_file).read_text().replace('\n', ',').split(','):
            tok = tok.strip()
            if not tok:
                continue
            if ':' not in tok:
                sys.exit('selector %r is not leg:num' % tok)
            (lg, _, num) = tok.partition(':')
            seeds.append(dict(leg=lg.strip(), seed=int(num), wpd_before=None))
        label = a.seeds_file
        cond = '(selector file; no condition recorded)'
    if not seeds:
        sys.exit('no seeds resolved')
    keys = [(r['leg'], r['seed']) for r in seeds]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        sys.exit('%d duplicated (leg,seed), e.g. %s' % (len(dups), dups[:3]))
    print('seed list %s   n=%d' % (label, len(seeds)))
    print('condition: %s' % cond)
    print('NOTE: distributions only. No geometric pass threshold is applied or implied.')
    print('NOTE: "FRJ-site" is the residue set the authors\' ligand FRJ contacts; our system holds 2e. Contact there is a classification, NOT pose retention.')
    rp = Path(a.ref).resolve()
    if not rp.is_file() or rp.stat().st_size == 0:
        sys.exit('REFUSING: --ref missing/empty: %s' % rp)
    rm = md5(rp)
    if rm != REF_MD5:
        msg = '--ref %s has md5 %s, not the pinned authors closed.pdb %s' % (rp, rm, REF_MD5)
        if not a.allow_unpinned_ref:
            sys.exit('REFUSING: ' + msg + '. Pass --allow-unpinned-ref to proceed deliberately; an earlier version only warned and carried on.')
        print('WARNING (explicitly allowed): ' + msg)
    print('reference %s  md5 %s%s' % (rp, rm, '' if rm == REF_MD5 else '  *** UNPINNED ***'))
    import numpy as np, pytraj as pt, parmed as pmd
    from scipy.spatial import cKDTree
    SH = [np.array([i, j, k], dtype=float) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    ref = pt.load(str(rp))
    rt = ref.top
    rF = rt.select(FIT_MASK)
    rW = rt.select(WPD_MASK)
    if len(rF) == 0 or len(rW) == 0:
        sys.exit('the reference selected no fit or WPD atoms')
    rFsig = topo_sig(rt, rF)
    rWsig = topo_sig(rt, rW)
    before = {}
    if a.strata:
        S = json.loads(Path(a.strata).read_text())
        for L in S.get('legs', {}).values():
            for r in L.get('rows', []):
                before[r['leg'], r['seed']] = r
        print('strata %s  rows %d' % (a.strata, len(before)))
    POSE = {}
    if a.pose_ref_root:
        root = Path(a.pose_ref_root).resolve()
        for leg in sorted({s['leg'] for s in seeds}):
            d = root / ('2e__%s' % leg)
            for (nm, pf, cf) in (('docked', 'dry.prmtop', 'dry.inpcrd'), ('equilibrated', 'system.prmtop', 'equil.rst7')):
                (pp, cc) = (d / pf, d / cf)
                if not (pp.is_file() and cc.is_file()):
                    sys.exit('REFUSING: pose reference %s for leg %s is missing (%s, %s)' % (nm, leg, pp, cc))
                t = pt.load(str(cc), top=str(pp))
                POSE[leg, nm] = t
        print('pose references loaded for %s' % sorted({l for (l, _) in POSE}))
    else:
        print('pose: NOT EVALUATED (--pose-ref-root not given)')
    root = Path(a.seeds_root).resolve()
    rows = []
    failed = []
    unverified_snaps = []
    unverified_inputs = []
    for s in seeds:
        (leg, n) = (s['leg'], s['seed'])
        d = root / leg / ('snapshot_%d' % n)
        why = []
        pd = d / 'PREP_DONE'
        if not (d.is_dir() and pd.is_file()):
            failed.append((leg, n, 'no PREP_DONE at %s' % d))
            continue
        (sig, outs, snap, snapmd5, sigtok) = sig_of(pd)
        if not sig:
            why.append('PREP_DONE has no signature line')
        for f in OUTS:
            fp = d / f
            if not fp.is_file():
                why.append('%s missing' % f)
                continue
            if f not in outs:
                why.append('PREP_DONE records no md5 for %s' % f)
                continue
            if md5(fp) != outs[f]:
                why.append('%s md5 differs from PREP_DONE' % f)
        in_state = {}
        for (k, fp) in expected_inputs(a.inputs_root, leg).items():
            rec = sigtok.get(k)
            if rec is None:
                why.append('PREP_DONE records no %s md5' % k)
                in_state[k] = 'unrecorded'
                continue
            if not fp.is_file():
                in_state[k] = 'not verified: %s absent' % fp
                unverified_inputs.append((leg, n, k, str(fp)))
                continue
            got = md5(fp)
            if got != rec:
                why.append('%s md5 %s does not match %s (%s)' % (k, rec[:8], got[:8], fp))
                in_state[k] = 'mismatch'
            else:
                in_state[k] = 'verified'
        snap_state = 'no snapshot/md5 recorded in PREP_DONE'
        if snap and snapmd5:
            sp = Path(snap)
            if not sp.is_file():
                snap_state = 'not verified: %s is gone' % sp
                unverified_snaps.append((leg, n, str(sp)))
            elif md5(sp) != snapmd5:
                why.append('the snapshot %s no longer has the md5 the signature recorded' % sp.name)
                snap_state = 'mismatch'
            else:
                snap_state = 'verified'
        if why:
            failed.append((leg, n, '; '.join(why)))
            continue
        try:
            w = pmd.load_file(str(d / 'seed_equil.rst7'))
        except Exception as e:
            failed.append((leg, n, 'parmed could not read the restart: %s' % e))
            continue
        wc = np.asarray(w.coordinates, dtype=float)
        wv = None if w.velocities is None else np.asarray(w.velocities, dtype=float)
        wb = None if w.box is None else np.asarray(w.box, dtype=float)
        if not np.all(np.isfinite(wc)):
            why.append('non-finite coordinates in the wet restart')
        if wv is None:
            why.append('the restart carries no velocities')
        elif not np.all(np.isfinite(wv)):
            why.append('non-finite velocities')
        if wb is None or wb.size < 6 or (not np.all(np.isfinite(wb[:6]))):
            why.append('missing or non-finite box')
        elif float(wb[:3].min()) <= 0:
            why.append('non-positive box length')
        if why:
            failed.append((leg, n, '; '.join(why)))
            continue
        try:
            t = pt.load(str(d / 'seed_equil.rst7'), top=str(d / 'system.prmtop'))
            n_wet = int(t.top.n_atoms)
            t = pt.strip(t, STRIP)
        except Exception as e:
            failed.append((leg, n, 'pytraj could not read/strip: %s' % e))
            continue
        tp = t.top
        X = t.xyz[0]
        L = tp.select(LIG_MASK)
        P = tp.select(PROT_MASK)
        F = tp.select(FIT_MASK)
        W = tp.select(WPD_MASK)
        if len(L) == 0 or len(P) == 0:
            why.append('ligand or protein mask selected nothing')
        if topo_sig(tp, F) != rFsig:
            why.append('fit-mask atoms do not correspond to the reference in order')
        if topo_sig(tp, W) != rWsig:
            why.append('WPD-mask atoms do not correspond to the reference in order')
        if why:
            failed.append((leg, n, '; '.join(why)))
            continue
        bi = np.array([[b.indices[0], b.indices[1]] for b in tp.bonds])
        bl = np.linalg.norm(X[bi[:, 0]] - X[bi[:, 1]], axis=1)
        maxbond = float(bl.max())
        coherent = bool(maxbond <= a.bond_max)
        if not coherent:
            failed.append((leg, n, 'solute is not a coherent molecular image: longest bonded distance %.2f A exceeds --bond-max %.2f, so the periodic image metric is not interpretable' % (maxbond, a.bond_max)))
            continue
        box = np.asarray(t[0].box.values if hasattr(t[0].box, 'values') else t[0].box, dtype=float)
        Mc = h3x.cell_vectors(box)
        sol = np.concatenate([tp.select(PROT_ALL), tp.select(LIG_ALL)])
        tree = cKDTree(X[sol])
        gap = float('inf')
        for sh in SH:
            (dd, _) = tree.query(X[sol] + sh @ Mc, k=1)
            gap = min(gap, float(dd.min()))
        D = h3x.min_image_matrix(X[L], X[P], Mc)
        rid = np.array([tp.atom(i).resid + 1 for i in P])
        cres = set(rid[np.where(D <= CONTACT_A)[1]].tolist())
        (R, tr) = h4p.kabsch(X[F], ref.xyz[0][rF])
        wpd = h4p.rmsd(X[W] @ R + tr, ref.xyz[0][rW])
        if not (np.isfinite(gap) and np.isfinite(wpd) and np.isfinite(D.min())):
            failed.append((leg, n, 'a measurement came out non-finite'))
            continue
        b = before.get((leg, n), {})
        rec = dict(leg=leg, seed=n, atoms_wet=n_wet, atoms_dry=int(tp.n_atoms), pairs_after=int((D <= CONTACT_A).sum()), frj_after=len(cres & FRJ_SITE), min_after=float(D.min()), wpd_after=float(wpd), image_gap_A=gap, max_bond_A=maxbond, coherent=coherent, box=[float(x) for x in box[:6]], pairs_before=b.get('pairs'), frj_before=b.get('frj'), wpd_before=b.get('wpd') if b.get('wpd') is not None else s['wpd_before'])
        for nm in ('docked', 'equilibrated'):
            key = (leg, nm)
            if key not in POSE:
                continue
            rtr = POSE[key]
            Xr = rtr.xyz[0]
            if rtr.top.n_atoms != tp.n_atoms:
                st = pt.strip(rtr, STRIP)
                Xr = st.xyz[0]
                if st.top.n_atoms != tp.n_atoms:
                    failed.append((leg, n, 'pose reference %s has %d atoms after stripping, the seed has %d' % (nm, st.top.n_atoms, tp.n_atoms)))
                    break
            Dr = np.sqrt(((Xr[L][:, None, :] - Xr[P][None, :, :]) ** 2).sum(-1))
            refc = set(rid[np.where(Dr <= CONTACT_A)[1]].tolist())
            rec['snapshot_check'] = snap_state
            rec['input_checks'] = in_state
            rec[nm] = h4p.pose_metrics(X, Xr, F, L, LIG_ALL and tp.select(LIG_ALL), P, rid, refc)
        else:
            rows.append(rec)
    print('\n=== after preparation (seed_equil.rst7), measured on %d seed(s)' % len(rows))
    if rows:
        keep = [r for r in rows if r['frj_after'] > 0]
        print('  still contacting an FRJ-site residue: %d of %d' % (len(keep), len(rows)))
        print('  no protein contact at all:            %d of %d' % (sum((1 for r in rows if r['pairs_after'] == 0)), len(rows)))
        print('  coherent molecular image:             %d of %d (longest bond <= %.1f A)' % (sum((1 for r in rows if r['coherent'])), len(rows), a.bond_max))
        for (k, lab) in (('pairs_after', 'contact pairs'), ('frj_after', 'FRJ-site residues'), ('min_after', 'closest pair A'), ('wpd_after', 'WPD RMSD A'), ('image_gap_A', 'solute-to-image gap A'), ('max_bond_A', 'longest bond A'), ('atoms_wet', 'wet atom count')):
            q = quant([r[k] for r in rows])
            if q:
                print('  %-22s min %9.2f  p10 %9.2f  median %9.2f  p90 %9.2f  max %9.2f' % (lab, q['min'], q['p10'], q['median'], q['p90'], q['max']))
        print('\n=== change from the trajectory frame to the prepared structure')
        for (k, lab) in (('wpd', 'WPD RMSD A'), ('pairs', 'contact pairs'), ('frj', 'FRJ-site residues')):
            v = [r[k + '_after'] - r[k + '_before'] for r in rows if r.get(k + '_before') is not None and r.get(k + '_after') is not None]
            q = quant(v)
            if q:
                print('  d(%-18s) min %+9.2f  median %+9.2f  max %+9.2f  (n=%d)' % (lab, q['min'], q['median'], q['max'], q['n']))
            else:
                print('  d(%-18s) unavailable (no before values; pass --strata)' % lab)
        if any(('equilibrated' in r for r in rows)):
            print('\n=== ligand pose after preparation (site contact is NOT pose retention)')
            for nm in ('docked', 'equilibrated'):
                sub = [r[nm] for r in rows if nm in r]
                if not sub:
                    continue
                print('  vs %s (n=%d)' % (nm, len(sub)))
                for (k, lab) in (('lig_rmsd_A', 'ligand RMSD A'), ('centroid_A', 'centroid shift A'), ('kept_contacts', 'kept contacts'), ('internal_rmsd_A', 'ligand internal A'), ('core_fit_rmsd_A', 'core fit RMSD A')):
                    q = quant([x[k] for x in sub if x[k] == x[k]])
                    if q:
                        print('    %-20s min %7.2f  p10 %7.2f  median %7.2f  p90 %7.2f  max %7.2f' % (lab, q['min'], q['p10'], q['median'], q['p90'], q['max']))
        else:
            print('\n=== ligand pose after preparation: NOT EVALUATED (--pose-ref-root absent)')
        wa = [r['wpd_after'] for r in rows]
        wb = [r['wpd_before'] for r in rows if r['wpd_before'] is not None]
        sa = np.sort(np.asarray(wa))
        g = sa[1:] - sa[:-1] if len(sa) > 1 else np.array([0.0])
        print('\n=== WPD coverage of the prepared set')
        if wb:
            print('  before (as selected)  %.3f - %.3f A  (span %.3f)' % (min(wb), max(wb), max(wb) - min(wb)))
        print('  after  (prepared)     %.3f - %.3f A  (span %.3f), largest internal gap %.3f A' % (sa[0], sa[-1], sa[-1] - sa[0], float(g.max())))
        print('  A wider min/max is NOT evidence of better coverage: the same number of seeds over a wider range is sparser, and the widening came from equilibration moving structures, not from the selection.')
    print('\n=== provenance of the inputs (signature recheck)')
    print('  input files recomputed from %s' % Path(a.inputs_root).resolve())
    nsv = len(unverified_snaps)
    niv = len(unverified_inputs)
    print('  snapshot md5 verified:   %d of %d requested seed(s)' % (len(seeds) - nsv - len(failed), len(seeds)))
    if nsv:
        print('  NOT VERIFIED (snapshot file gone, so the restart has no checkable origin): %d' % nsv)
        for (lg, sd, sp) in unverified_snaps[:6]:
            print('    %-7s %3d  %s' % (lg, sd, sp))
        if nsv > 6:
            print('    ... and %d more' % (nsv - 6))
    if niv:
        print('  NOT VERIFIED (input file absent): %d seed-input pair(s)' % niv)
        seen = set()
        for (lg, sd, k, fp) in unverified_inputs:
            if (k, fp) not in seen:
                seen.add((k, fp))
                print('    %-7s %s' % (k, fp))
    if not nsv and (not niv) and (not failed):
        print('  every mol2/frcmod/parent/em/heat/npt md5 in every PREP_DONE was recomputed from the file on disk and matched.')
    print('\n=== completeness')
    if failed:
        print('  INCOMPLETE: %d of %d requested seed(s) did not pass checks 1-5' % (len(failed), len(seeds)))
        for (lg, sd, w) in failed[:12]:
            print('    %-7s %3d  %s' % (lg, sd, w))
        if len(failed) > 12:
            print('    ... and %d more' % (len(failed) - 12))
        print('  A partial comparison is not a comparison.')
    else:
        print('  all %d requested seed(s) present, readable and through checks 1-5' % len(seeds))
    print('\nNo verdict is given on any geometric quantity. Any threshold must be written down beforehand, elsewhere. Preparation completeness is not production authorisation.')
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(dict(seed_list=label, condition=cond, seeds_root=str(root), ref=str(rp), ref_md5=rm, strata=a.strata, pose_ref_root=a.pose_ref_root, bond_max_A=a.bond_max, cutoff_A=CONTACT_A, n_requested=len(seeds), n_measured=len(rows), failed=[dict(leg=l, seed=s_, why=w) for (l, s_, w) in failed], frames=rows), f, indent=1, ensure_ascii=False)
        print('record -> %s' % a.json)
    return 1 if failed else 0
if __name__ == '__main__':
    sys.exit(main())
