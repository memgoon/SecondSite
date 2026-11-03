# SPDX-License-Identifier: MIT
import argparse, datetime, hashlib, json, os, re, subprocess, sys
from pathlib import Path
FIT_MASK = ':1-282&(!@/H)&(!:198)'
WPD_MASK = ':179-185@CA,C,N,O'
STRIP = ':WAT,:SOL,:HOH,:NA,:CL,:Na+,:Cl-,:CLA,:POT,:SOD'
LIG_MASK = ':296&!@/H'
PROT_MASK = '(:1-295)&!@/H&!:WAT,Na+,Cl-'
ANCHOR = ':1-295'
FRJ_SITE = {189, 190, 193, 194, 197, 201, 277, 278, 280, 281}
DT_PS = 0.002
NTWX = 2500
CONTACT_A = 4.5
REF_MD5 = '1d9ad04e386db0298f4b94a95446d9d6'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda : f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def sh(cmd, cwd):
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True, text=True)

def need_cpptraj(binary):
    import shutil
    if not (os.path.isabs(str(binary)) and os.access(str(binary), os.X_OK)) and shutil.which(str(binary)) is None:
        sys.exit('cpptraj not found as %r. Run this inside the container, or pass --cpptraj / set CPPTRAJ_BIN.' % binary)

def run_cpptraj(binary, inp, cwd, outputs, what):
    for o in outputs:
        if Path(o).exists():
            sys.exit('REFUSING: %s already exists before %s ran. The scratch directory is meant to be fresh.' % (o, what))
    r = sh([binary, '-i', inp], cwd=cwd)
    if r.returncode != 0:
        sys.exit('cpptraj FAILED (%s, status %d):\n%s' % (what, r.returncode, r.stderr[-2000:]))
    for o in outputs:
        if not Path(o).exists() or Path(o).stat().st_size == 0:
            sys.exit('cpptraj reported success but %s is missing or empty (%s).\n%s' % (o, what, r.stdout[-2000:]))
    return r

def cell_vectors(box):
    import numpy as np
    (a, b, c, al, be, ga) = [float(x) for x in box[:6]]
    (al, be, ga) = [np.deg2rad(x) for x in (al, be, ga)]
    v1 = np.array([a, 0.0, 0.0])
    v2 = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / max(np.sin(ga), 1e-12)
    cz2 = c * c - cx * cx - cy * cy
    v3 = np.array([cx, cy, np.sqrt(max(cz2, 0.0))])
    return np.vstack([v1, v2, v3])

def perp_widths(M):
    import numpy as np
    V = abs(np.linalg.det(M))
    out = []
    for i in range(3):
        (j, k) = [x for x in range(3) if x != i]
        out.append(V / np.linalg.norm(np.cross(M[j], M[k])))
    return np.array(out)
SHIFTS27 = None

def _shifts(n):
    import numpy as np
    return np.array([[i, j, k] for i in range(-n, n + 1) for j in range(-n, n + 1) for k in range(-n, n + 1)], dtype=float)

def min_image_matrix(X, Y, M, Minv=None):
    import numpy as np
    if Minv is None:
        Minv = np.linalg.inv(M)
    D = X[:, None, :] - Y[None, :, :]
    F = D @ Minv
    D = (F - np.round(F)) @ M
    best = None
    for sh in _shifts(1):
        d = np.sqrt(((D + sh @ M) ** 2).sum(-1))
        best = d if best is None else np.minimum(best, d)
    return best

def min_image_bruteforce(X, Y, M, ncell=3, Minv=None, cap=5):
    import numpy as np
    if Minv is None:
        Minv = np.linalg.inv(M)
    F = (X[:, None, :] - Y[None, :, :]) @ Minv
    needed = int(np.ceil(np.abs(F).max())) + 1
    n = max(int(ncell), needed)
    if n > cap:
        return (None, needed)
    best = None
    for sh in _shifts(n):
        d = np.sqrt((((X + sh @ M)[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
        best = d if best is None else np.minimum(best, d)
    return (best, n)

def refinement_is_sufficient(X, Y, M, Minv=None, wide=2):
    import numpy as np
    if Minv is None:
        Minv = np.linalg.inv(M)
    D = X[:, None, :] - Y[None, :, :]
    F = D @ Minv
    D = (F - np.round(F)) @ M
    narrow = None
    broad = None
    for sh in _shifts(wide):
        d = np.sqrt(((D + sh @ M) ** 2).sum(-1))
        broad = d if broad is None else np.minimum(broad, d)
        if np.abs(sh).max() <= 1:
            narrow = d if narrow is None else np.minimum(narrow, d)
    return (narrow, broad)

def runs_of_zero(rows, key, expect):
    out = []
    s = None
    prev = None
    for r in rows:
        if prev is not None and r['step'] - prev['step'] > expect and (s is not None):
            out.append((s, prev['ns'], 'ended at a GAP'))
            s = None
        if r[key] == 0 and s is None:
            s = r['ns']
        elif r[key] != 0 and s is not None:
            out.append((s, r['ns'], 'recovered'))
            s = None
        prev = r
    if s is not None:
        out.append((s, rows[-1]['ns'], 'still zero at the last frame'))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='<W>/<state>/production')
    ap.add_argument('--ref', required=True, help='the authors closed.pdb')
    ap.add_argument('--scratch', required=True, help='where cpptraj may write (NOT --dir)')
    ap.add_argument('--cpptraj', default=os.environ.get('CPPTRAJ_BIN') or 'cpptraj')
    ap.add_argument('--stride', type=int, default=10, help='written frames to skip (1 frame = 5 ps). Default 10 = 50 ps')
    ap.add_argument('--from-ns', type=float, default=None, help='ignore frames before this')
    ap.add_argument('--to-ns', type=float, default=None, help='ignore frames after this')
    ap.add_argument('--print-every', type=float, default=1.0, help='ns between printed rows')
    ap.add_argument('--allow-running', action='store_true', help='also read the live segment, limited to the last NSTEP its .out reports. Without this, only segments with a recorded valid length are read.')
    ap.add_argument('--steps-json', default=None, help='an h4c SEEDS.json: analyse EXACTLY the frames at its chosen[].absolute_step, which is what a definitive per-seed tally requires. Overrides --from-ns/--to-ns/--stride.')
    ap.add_argument('--steps', default=None, help='comma-separated absolute steps to analyse exactly')
    ap.add_argument('--selfcheck-frames', type=int, default=5, help='cross-validate the minimum image against an independent wide cell search on this many frames (0 disables)')
    ap.add_argument('--selfcheck-cells', type=int, default=3, help='MINIMUM whole-cell range for the reduction-free cross-check; it is widened automatically when the frame needs more')
    ap.add_argument('--selfcheck-cap', type=int, default=5, help='largest whole-cell range the cross-check will run (cost grows as (2n+1)^3). Beyond it the check is SKIPPED, never run narrow.')
    ap.add_argument('--json', default=None, help='write the full series here')
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    ref = Path(a.ref).resolve()
    scr0 = Path(a.scratch).resolve()
    if scr0 == d or d in scr0.parents or (scr0 in [d, *d.parents] and scr0 == d):
        sys.exit('REFUSING: --scratch is the production directory.')
    if d == scr0 or str(scr0).startswith(str(d) + os.sep):
        sys.exit('REFUSING: --scratch is inside the production directory (%s). This tool must not write there.' % d)
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    scr = scr0 / ('run_' + stamp)
    scr.mkdir(parents=True, exist_ok=False)
    if not ref.is_file() or ref.stat().st_size == 0:
        sys.exit('REFUSING: --ref is missing or empty: %s' % ref)
    rm = md5(ref)
    print('h3x ligand-escape diagnostic   scratch %s' % scr)
    print('reference %s  md5 %s%s' % (ref, rm, '' if rm == REF_MD5 else '  *** NOT the pinned authors closed.pdb (%s) ***' % REF_MD5))
    need_cpptraj(a.cpptraj)
    want_steps = None
    if a.steps_json:
        sj = json.loads(Path(a.steps_json).read_text())
        want_steps = sorted({int(c['absolute_step']) for c in sj.get('chosen', [])})
        if not want_steps:
            sys.exit('%s has no chosen[].absolute_step' % a.steps_json)
        print('exact-step mode: %d step(s) from %s' % (len(want_steps), a.steps_json))
    elif a.steps:
        want_steps = sorted({int(x) for x in a.steps.replace(',', ' ').split()})
        print('exact-step mode: %d step(s) from --steps' % len(want_steps))
    pj = d / 'PROGRESS.json'
    if not pj.exists():
        sys.exit('no PROGRESS.json in %s' % d)
    prog = json.loads(pj.read_text())
    state = prog.get('state', '?')
    segs = sorted(prog.get('segments', []), key=lambda x: x['segment'])
    iwrap = set()
    for s in segs:
        o = d / s.get('files', {}).get('out', 'seg_%s.out' % s['segment'])
        if o.exists():
            for m in re.finditer('iwrap\\s*=\\s*(\\d+)', o.read_text(errors='replace')):
                iwrap.add(int(m.group(1)))
    print('iwrap as echoed by Amber: %s' % (sorted(iwrap) if iwrap else 'NOT FOUND in the .out namelist echo - treat wrapping as unknown'))
    print('NOTE: iwrap only says whether coordinates were wrapped. It does not make a raw Cartesian distance a minimum-image distance, which is why all three distances below are computed.')
    try:
        import numpy as np, pytraj as pt
    except Exception as e:
        sys.exit('needs numpy and pytraj (run inside the container): %s' % e)
    plan = []
    for s in segs:
        tag = s['segment']
        nc = d / ('seg_%s.nc' % tag)
        limit = s.get('valid_to_step')
        src = 'valid_to_step'
        if limit is None:
            limit = s.get('end_step')
            src = 'end_step'
        if limit is None:
            if not a.allow_running:
                print('  seg_%s (status %r): no recorded valid length; SKIPPED (pass --allow-running to read the live segment)' % (tag, s.get('status')))
                continue
            o = d / s.get('files', {}).get('out', 'seg_%s.out' % tag)
            hits = re.findall('NSTEP =\\s*(\\d+)', o.read_text(errors='replace')) if o.exists() else []
            if not hits:
                print('  seg_%s: live but its .out reports no NSTEP; SKIPPED' % tag)
                continue
            limit = s['start_step'] + int(hits[-1])
            src = 'live_nstep'
        if not nc.exists() or nc.stat().st_size < 10000:
            print('  seg_%s: no usable trajectory on disk; SKIPPED' % tag)
            continue
        nframe = (int(limit) - int(s['start_step'])) // NTWX
        if nframe < 1:
            print('  seg_%s: 0 complete frames within %s=%d; SKIPPED' % (tag, src, limit))
            continue
        plan.append((tag, int(s['start_step']), int(limit), nframe, src))
        print('  seg_%s: %d frames usable, limit %s=%d' % (tag, nframe, src, limit))
    if not plan:
        sys.exit('no segment had a usable, validated trajectory')
    print('\nstate=%s  cutoff %.1f A  stride %d (%.3f ns between frames)' % (state, CONTACT_A, a.stride, a.stride * NTWX * DT_PS / 1000.0))
    rows = []
    warn_cut = 0
    checked = 0
    selfcheck_skipped = []
    for (tag, start, limit, nframe, src) in plan:
        nc = d / ('seg_%s.nc' % tag)
        raw = scr / ('raw_%s.nc' % tag)
        img = scr / ('img_%s.nc' % tag)
        wpdf = scr / ('wpd_%s.dat' % tag)
        if want_steps is None:
            trajin = ['trajin %s 1 %d %d' % (nc, nframe, a.stride)]
            nsel = (nframe - 1) // a.stride + 1
            frame_steps = [start + (1 + i * a.stride) * NTWX for i in range(nsel)]
        else:
            trajin = []
            frame_steps = []
            for st_ in want_steps:
                off = st_ - start
                if off <= 0 or off % NTWX:
                    continue
                fr_ = off // NTWX
                if fr_ < 1 or fr_ > nframe:
                    continue
                trajin.append('trajin %s %d %d 1' % (nc, fr_, fr_))
                frame_steps.append(st_)
            if not trajin:
                print('  seg_%s: none of the requested steps fall in it' % tag)
                continue
            print('  seg_%s: %d requested step(s) read exactly' % (tag, len(trajin)))
        (scr / ('a_%s.in' % tag)).write_text('\n'.join(['parm %s' % (d / 'system.prmtop')] + trajin + ['strip %s' % STRIP, 'trajout %s netcdf' % raw, 'go', 'quit', '']))
        run_cpptraj(a.cpptraj, scr / ('a_%s.in' % tag), scr, [raw], 'raw pass on seg_%s' % tag)
        (scr / ('b_%s.in' % tag)).write_text('\n'.join(['parm %s' % (d / 'system.prmtop')] + trajin + ['autoimage anchor %s' % ANCHOR, 'strip %s' % STRIP, 'trajout %s netcdf' % img, 'go', 'quit', '']))
        run_cpptraj(a.cpptraj, scr / ('b_%s.in' % tag), scr, [img], 'autoimage pass on seg_%s' % tag)
        (scr / ('c_%s.in' % tag)).write_text('\n'.join(['parm %s' % (d / 'system.prmtop')] + trajin + ['strip %s' % STRIP, 'parm %s name refp' % ref, 'reference %s parm refp name cref' % ref, 'rms %s ref cref' % FIT_MASK, 'rmsd %s out %s nofit ref cref' % (WPD_MASK, wpdf), 'go', 'quit', '']))
        run_cpptraj(a.cpptraj, scr / ('c_%s.in' % tag), scr, [wpdf], 'WPD pass on seg_%s' % tag)
        wpd = []
        for l in wpdf.read_text().splitlines():
            if l.startswith('#'):
                continue
            p = l.split()
            if len(p) >= 2:
                wpd.append(float(p[1]))
        dry = scr / 'dry.prmtop'
        if not dry.exists():
            (scr / 'dry.in').write_text('parm %s\nparmstrip %s\nparmwrite out %s\ngo\nquit\n' % (d / 'system.prmtop', STRIP, dry))
            run_cpptraj(a.cpptraj, scr / 'dry.in', scr, [dry], 'stripped topology')
        tr = pt.iterload(str(raw), top=str(dry))
        ti = pt.iterload(str(img), top=str(dry))
        if tr.n_frames != ti.n_frames:
            sys.exit('raw and imaged passes disagree on frame count for seg_%s (%d vs %d)' % (tag, tr.n_frames, ti.n_frames))
        tp = tr.top
        L = tp.select(LIG_MASK)
        P = tp.select(PROT_MASK)
        if len(L) == 0 or len(P) == 0:
            sys.exit('ligand or protein mask selected nothing')
        rid = np.array([tp.atom(i).resid + 1 for i in P])
        for (i, (fr, fi)) in enumerate(zip(tr, ti)):
            step = frame_steps[i] if i < len(frame_steps) else start + (1 + i * a.stride) * NTWX
            ns = step * DT_PS / 1000.0
            if want_steps is None:
                if a.from_ns is not None and ns < a.from_ns:
                    continue
                if a.to_ns is not None and ns > a.to_ns:
                    continue
            X = fr.xyz[L]
            Y = fr.xyz[P]
            Draw = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
            Xi = fi.xyz[L]
            Yi = fi.xyz[P]
            Dimg = np.sqrt(((Xi[:, None, :] - Yi[None, :, :]) ** 2).sum(-1))
            box = np.asarray(fr.box.values if hasattr(fr.box, 'values') else fr.box, dtype=float)
            M = cell_vectors(box)
            widths = perp_widths(M)
            if CONTACT_A > 0.5 * widths.min():
                warn_cut += 1
            best = min_image_matrix(X, Y, M)
            if checked < a.selfcheck_frames:
                (nar, bro) = refinement_is_sufficient(X, Y, M)
                if not np.allclose(nar, bro, atol=1e-06):
                    k = int(np.argmax(np.abs(nar - bro)))
                    sys.exit('BUG: widening the neighbour refinement finds a closer image at %.3f ns (%.4f vs %.4f A), so a +-1 refinement is not sufficient for this cell. No conclusion may be drawn.' % (ns, nar.flat[k], bro.flat[k]))
                (bf, used) = min_image_bruteforce(X, Y, M, a.selfcheck_cells, cap=a.selfcheck_cap)
                if bf is None:
                    selfcheck_skipped.append((ns, used))
                elif not np.allclose(best, bf, atol=1e-06):
                    k = int(np.argmax(np.abs(best - bf)))
                    sys.exit('BUG: the reduced minimum image disagrees with an independent +-%d-cell reduction-free search at %.3f ns (worst pair %.4f vs %.4f A). No conclusion may be drawn from this run.' % (used, ns, best.flat[k], bf.flat[k]))
                checked += 1
            if best.min() > Draw.min() + 1e-06:
                sys.exit('BUG: the minimum image (%.4f A) exceeds the raw distance (%.4f A) at %.3f ns. The cell arithmetic is wrong; no conclusion may be drawn from this run.' % (best.min(), Draw.min(), ns))
            cres_i = set(rid[np.where(Dimg <= CONTACT_A)[1]].tolist())
            cres_m = set(rid[np.where(best <= CONTACT_A)[1]].tolist())
            rows.append(dict(ns=ns, step=int(step), seg=tag, exact=bool(want_steps), raw_pairs=int((Draw <= CONTACT_A).sum()), raw_min=float(Draw.min()), img_pairs=int((Dimg <= CONTACT_A).sum()), img_min=float(Dimg.min()), mim_pairs=int((best <= CONTACT_A).sum()), mim_min=float(best.min()), img_frj=len(cres_i & FRJ_SITE), img_res=len(cres_i), mim_frj=len(cres_m & FRJ_SITE), mim_res=len(cres_m), wpd=wpd[i] if i < len(wpd) else float('nan'), box_min_width=float(widths.min())))
    if not rows:
        sys.exit('no frames fell inside the requested window')
    rows.sort(key=lambda r: r['ns'])
    steps = [r['step'] for r in rows]
    dup = sorted({s for s in steps if steps.count(s) > 1})
    expect = a.stride * NTWX
    gaps = []
    for (p, q) in zip(rows, rows[1:]):
        if q['step'] - p['step'] > expect:
            gaps.append((p['ns'], q['ns'], p['seg'], q['seg']))
    if dup:
        print('\nWARNING: %d duplicated absolute step(s), e.g. %s' % (len(dup), dup[:5]))
    if want_steps is not None:
        print('\nexact-step mode: the %d requested frames are not contiguous (they are the selected seeds, about %.2f ns apart), so zero-contact INTERVALS are not a defined quantity here and are not reported. Each frame stands alone; see the per-frame table and the state counts.' % (len(rows), (rows[-1]['ns'] - rows[0]['ns']) / max(1, len(rows) - 1)))
    elif gaps:
        print('\n%d gap(s) in the time axis - a zero-contact run is NEVER continued across one:' % len(gaps))
        for (x, y, sa, sb) in gaps[:10]:
            print('  %.3f -> %.3f ns  (seg_%s -> seg_%s)' % (x, y, sa, sb))
    if selfcheck_skipped:
        print('\nNOTE: the reduction-free cross-check was skipped on %d frame(s) because the separation spanned more cells than --selfcheck-cap=%d allows:' % (len(selfcheck_skipped), a.selfcheck_cap))
        for (ns_, need) in selfcheck_skipped[:5]:
            print('  %.3f ns needed a +-%d-cell search' % (ns_, need))
        print('  This is a limit of the CHECK, not a fault in the result, and it does not mean the ligand left: a large recorded separation is expected with iwrap=0. The refinement-sufficiency check above still ran on those frames.')
    if warn_cut:
        print('\nWARNING: on %d frame(s) the %.1f A cutoff exceeds half the smallest box width, so the 27-cell search may not be exact there.' % (warn_cut, CONTACT_A))
    print('\n%8s %5s|%6s %7s|%6s %7s %4s|%6s %7s %4s|%8s' % ('ns', 'seg', 'rawP', 'rawMin', 'imgP', 'imgMin', 'FRJ', 'mimP', 'mimMin', 'FRJ', 'wpdRMSD'))
    nxt = 0.0
    for r in rows:
        if r['ns'] >= nxt:
            print('%8.2f %5s|%6d %7.2f|%6d %7.2f %4d|%6d %7.2f %4d|%8.3f' % (r['ns'], r['seg'], r['raw_pairs'], r['raw_min'], r['img_pairs'], r['img_min'], r['img_frj'], r['mim_pairs'], r['mim_min'], r['mim_frj'], r['wpd']))
            nxt = r['ns'] + a.print_every
    r = rows[-1]
    print('%8.2f %5s|%6d %7.2f|%6d %7.2f %4d|%6d %7.2f %4d|%8.3f   <- last' % (r['ns'], r['seg'], r['raw_pairs'], r['raw_min'], r['img_pairs'], r['img_min'], r['img_frj'], r['mim_pairs'], r['mim_min'], r['mim_frj'], r['wpd']))
    dis = [r for r in rows if (r['raw_pairs'] == 0) != (r['mim_pairs'] == 0)]
    print('\nagreement between the raw metric and the exact minimum image:')
    print('  frames analysed            %d' % len(rows))
    print('  raw says zero contacts     %d' % sum((1 for r in rows if r['raw_pairs'] == 0)))
    print('  autoimage says zero        %d' % sum((1 for r in rows if r['img_pairs'] == 0)))
    print('  minimum image says zero    %d' % sum((1 for r in rows if r['mim_pairs'] == 0)))
    print('  raw and minimum image DISAGREE on %d frame(s)%s' % (len(dis), ' <-- the raw metric, and the monitor FAIL built on it, is an imaging artifact on those frames' if dis else ' - the departure is not an imaging artifact'))
    da = [r for r in rows if (r['img_pairs'] == 0) != (r['mim_pairs'] == 0)]
    print('\nagreement between autoimage-anchored and the exact minimum image:')
    print('  they DISAGREE on zero/non-zero contact on %d frame(s)' % len(da))
    if da:
        print('  %8s %7s %7s %7s %7s %5s %5s' % ('ns', 'imgP', 'imgMin', 'mimP', 'mimMin', 'iFRJ', 'mFRJ'))
        for r in da[:20]:
            print('  %8.2f %7d %7.2f %7d %7.2f %5d %5d' % (r['ns'], r['img_pairs'], r['img_min'], r['mim_pairs'], r['mim_min'], r['img_frj'], r['mim_frj']))
        nf = sum((1 for r in da if r['mim_frj'] > 0))
        print('  of these, %d contact at least one FRJ-site residue under the minimum image, i.e. the disagreement is about the ORIGINAL pocket; the other %d are contact elsewhere on the surface.' % (nf, len(da) - nf))
        print('  likely cause: autoimage anchors on the centre of %s, so a ligand near a protein edge can be imaged away from its nearest atom pair. The minimum image is the metric to trust here.' % ANCHOR)
    if want_steps is None:
        for (key, label) in (('raw_pairs', 'raw'), ('mim_pairs', 'minimum image')):
            rr = runs_of_zero(rows, key, expect)
            print('\nzero-contact intervals, %s metric:' % label)
            if not rr:
                print('  none')
            for (x, y, how) in rr:
                print('  %7.2f -> %7.2f ns  (%.2f ns, %s)' % (x, y, y - x, how))
    else:
        for (key, label) in (('raw_pairs', 'raw'), ('mim_pairs', 'minimum image')):
            n0 = sum((1 for r in rows if r[key] == 0))
            print('\nzero-contact FRAMES, %s metric: %d of %d' % (label, n0, len(rows)))
    far = max(rows, key=lambda r: r['mim_min'])
    print('\nfarthest, by exact minimum image: %.2f A at %.2f ns' % (far['mim_min'], far['ns']))
    inpocket = [r for r in rows if r['mim_frj'] > 0]
    print('frames with at least one FRJ-site residue contacted (minimum image): %d of %d' % (len(inpocket), len(rows)))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(state=state, dir=str(d), cutoff_A=CONTACT_A, ref_md5=rm, iwrap_echoed=sorted(iwrap), stride=a.stride, scratch=str(scr), duplicated_steps=dup, gaps=[dict(from_ns=x, to_ns=y, from_seg=sa, to_seg=sb) for (x, y, sa, sb) in gaps], frames=rows), indent=1, ensure_ascii=False))
        print('\nfull series -> %s' % a.json)
    return 0
if __name__ == '__main__':
    sys.exit(main())
