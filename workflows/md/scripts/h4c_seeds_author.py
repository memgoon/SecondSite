# SPDX-License-Identifier: MIT
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
CACHE = Path(os.environ.get('AMMO_REFERENCE_DIR', str(Path(__file__).resolve().parents[1] / 'refs')))
REF_NAME = 'ammo_examples__example_ptp1b__inputs__closed.pdb'

def find_ref(explicit):
    here = Path(__file__).resolve().parent
    if explicit:
        c = Path(explicit).resolve()
        return c if c.exists() and c.stat().st_size > 0 else None
    for c in (here.parent / 'refs' / 'closed.pdb', here / 'refs' / 'closed.pdb', CACHE / REF_NAME):
        if c.exists() and c.stat().st_size > 0:
            return c.resolve()
    return None
FIT_MASK = ':1-282&(!@/H)&(!:198)'
WPD_MASK = ':179-185@CA,C,N,O'
STRIP = ':WAT,:SOL,:HOH,:NA,:CL,:Na+,:Cl-,:CLA,:POT,:SOD'
DT_PS = 0.002
N_SEEDS = 100

def require(x, m):
    if not x:
        raise SystemExit('h4c: ' + m)

def cpptraj(lines, cwd, log):
    inp = Path(cwd) / ('_%s.in' % log)
    inp.write_text('\n'.join(lines) + '\ngo\nquit\n')
    r = subprocess.run([TRAJ, '-i', str(inp)], cwd=str(cwd), capture_output=True, text=True)
    (Path(cwd) / ('_%s.log' % log)).write_text(r.stdout + r.stderr)
    require(r.returncode == 0 and 'Error' not in r.stderr, 'cpptraj failed (%s); see _%s.log\n%s' % (log, log, r.stderr[-800:]))
    return r.stdout

def main():
    global TRAJ
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a finished sMD run directory')
    ap.add_argument('--out', default=None, help='default <run>/snapshots')
    ap.add_argument('--ntwx', type=int, default=2500, help='trajectory stride used by the mdin')
    ap.add_argument('--n', type=int, default=N_SEEDS)
    ap.add_argument('--mode', choices=('authors', 'rmsd_equal'), default='authors')
    ap.add_argument('--duration-ns', type=float, default=None, help="steered portion to sample; default is the notebook's floor(last_frame_ns/10)*10")
    ap.add_argument('--ref', default=None, help="the authors' inputs/closed.pdb; default searches <package>/refs/closed.pdb then AMMO_REFERENCE_DIR")
    ap.add_argument('--scratch', default=None, help='where cpptraj input/log/dat files go. Default <out>/_scratch. NEVER the run directory: the run must stay untouched.')
    ap.add_argument('--cpptraj', default=os.environ.get('CPPTRAJ_BIN') or shutil.which('cpptraj'))
    ap.add_argument('--allow-partial', action='store_true', help='DEVIATION: proceed on an unfinished or gapped run. The reproduction default refuses, because substituting a neighbouring frame for a missing one is not the authors selection and would be invisible in the output.')
    a = ap.parse_args()
    TRAJ = a.cpptraj
    require(TRAJ, 'cpptraj not found; set --cpptraj or CPPTRAJ_BIN')
    run = Path(a.run).resolve()
    out = Path(a.out) if a.out else run / 'snapshots'
    require((run / 'PROGRESS.json').exists(), 'no PROGRESS.json in ' + str(run))
    require((run / 'system.prmtop').exists(), 'no system.prmtop in ' + str(run))
    ref = find_ref(a.ref)
    require(ref is not None, "the authors' closed.pdb was not found. Pass --ref, or put it at <package>/refs/closed.pdb. Measuring against the wrong reference would be worse than failing.")
    scratch = Path(a.scratch).resolve() if a.scratch else out / '_scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    require(scratch.resolve() != run.resolve(), 'the scratch directory must not be the run directory - the run has to stay untouched')
    print('  reference %s' % ref)
    print('  scratch   %s' % scratch)
    if not a.allow_partial:
        require((run / 'H3_DONE').exists(), 'the run carries no H3_DONE, so the sMD is not finished. Pass --allow-partial only if you intend a deliberate deviation.')
    prog = json.loads((run / 'PROGRESS.json').read_text())
    segs = sorted(prog['segments'], key=lambda x: x['segment'])
    require(segs, 'PROGRESS.json lists no segments')
    out.mkdir(parents=True, exist_ok=True)
    stride_ns = a.ntwx * DT_PS / 1000.0
    index = []
    missing = []
    limit_sources = {}
    for s in segs:
        nc = run / ('seg_%s.nc' % s['segment'])
        if not nc.exists():
            require(a.allow_partial, 'PROGRESS.json lists segment %s but %s is missing. Refusing to seed from an incomplete trajectory.' % (s['segment'], nc.name))
            print('  seg_%s: %s MISSING, skipped (--allow-partial)' % (s['segment'], nc.name))
            missing.append(s['segment'])
            continue
        dat = scratch / ('_wpd_%s.dat' % s['segment'])
        cpptraj(['parm %s' % (run / 'system.prmtop'), 'trajin %s' % nc, 'parm %s name refp' % ref, 'reference %s parm refp name cref' % ref, 'rms %s ref cref' % FIT_MASK, 'rmsd %s out %s nofit ref cref' % (WPD_MASK, dat)], scratch, 'wpd_%s' % s['segment'])
        vals = []
        for l in dat.read_text().splitlines():
            if l.startswith('#'):
                continue
            p = l.split()
            if len(p) >= 2:
                vals.append(float(p[1]))
        require(vals, 'no RMSD values for seg_%s' % s['segment'])
        limit = s.get('valid_to_step')
        limit_src = 'valid_to_step'
        if limit is None:
            limit = s.get('end_step')
            limit_src = 'end_step'
        if limit is None and a.allow_partial:
            limit = s.get('planned_end_step')
            limit_src = 'planned_end_step'
            if limit is not None:
                print('  seg_%s: no completed length recorded; using planned_end_step=%d (DEVIATION, --allow-partial)' % (s['segment'], limit))
        require(limit is not None, 'segment %s (status %r) records no completed length - it has neither valid_to_step nor end_step. A running segment carries only planned_end_step, which is an intention, not frames on disk, so it cannot be seeded from in the reproduction path. Wait for H3 to finish, or pass --allow-partial to accept the planned length as a deliberate deviation.' % (s['segment'], s.get('status')))
        limit_sources[s['segment']] = limit_src
        kept = 0
        for (i, v) in enumerate(vals):
            abs_step = s['start_step'] + (i + 1) * a.ntwx
            if abs_step > limit:
                continue
            index.append((abs_step, v, s['segment'], i + 1))
            kept += 1
        print('  seg_%s[%s]: %d frames, %d within valid_to_step=%d' % (s['segment'], s.get('status', '?'), len(vals), kept, limit))
    require(index, 'no usable frames found')
    index.sort(key=lambda x: x[0])
    steps = [x[0] for x in index]
    require(len(set(steps)) == len(steps), 'two frames carry the same absolute step; the segment bookkeeping is wrong')
    last_ns = steps[-1] * DT_PS / 1000.0
    lo = min((x[1] for x in index))
    hi = max((x[1] for x in index))
    print('  %d frames, %.3f - %.3f ns, WPD RMSD to closed %.3f - %.3f A' % (len(index), steps[0] * DT_PS / 1000.0, last_ns, lo, hi))
    expected = [(i + 1) * a.ntwx for i in range(len(index))]
    gapless = steps == expected
    if not gapless:
        bad = next((i for i in range(len(steps)) if steps[i] != expected[i]))
        require(a.allow_partial, 'the concatenated trajectory is not the gapless series the authors index: frame %d sits at step %d, not %d. A crashed or truncated segment is the usual cause. Refusing to substitute neighbouring frames.' % (bad, steps[bad], expected[bad]))
        print('  WARNING: trajectory has gaps from frame %d (--allow-partial)' % bad)
    if a.mode == 'authors':
        dur = a.duration_ns if a.duration_ns is not None else int(last_ns // 10) * 10
        require(dur > 0, 'the steered portion rounds down to 0 ns; pass --duration-ns')
        stride = index[1][0] - index[0][0] if len(index) > 1 else a.ntwx
        require(stride == a.ntwx, 'consecutive frames are %d steps apart but --ntwx says %d' % (stride, a.ntwx))
        last_frame = int(dur / stride_ns)
        picks = [int(x) for x in np.linspace(0, last_frame, a.n, dtype=int)]
        require(len(set(picks)) == a.n, 'np.linspace(0, %d, %d) gives only %d distinct frames' % (last_frame, a.n, len(set(picks))))
        if gapless:
            require(max(picks) <= len(index) - 1, 'the selection needs trajectory frame %d (%.3f ns) but the run only wrote %d frames (%.3f ns)' % (max(picks), (max(picks) + 1) * stride_ns, len(index), last_ns))
            chosen = picks
            sel = 'np.linspace(0, %d, %d, dtype=int) over the first %g ns, positional, exact (authors sMD_analysis.ipynb cell 15)' % (last_frame, a.n, dur)
            drift = 0
        else:
            want = [(p + 1) * a.ntwx for p in picks]
            used = set()
            chosen = []
            for w in want:
                for k in sorted(range(len(index)), key=lambda k: (abs(steps[k] - w), steps[k])):
                    if k not in used:
                        used.add(k)
                        chosen.append(k)
                        break
            require(len(chosen) == a.n, 'picked %d of %d seeds' % (len(chosen), a.n))
            chosen.sort(key=lambda k: steps[k])
            drift = max((abs(steps[k] - w) for (k, w) in zip(chosen, sorted(want))))
            print('  DEVIATION: gapped trajectory, nearest-frame substitution used; worst offset %d steps (%.3f ns)' % (drift, drift * DT_PS / 1000.0))
            sel = 'np.linspace(0, %d, %d, dtype=int) over the first %g ns, with nearest-frame substitution because the trajectory has gaps (DEVIATION)' % (last_frame, a.n, dur)
    else:
        require(hi > lo, 'the WPD RMSD range is degenerate; nothing to sample across')
        dur = None
        drift = None
        want = [lo + (hi - lo) * i / (a.n - 1) for i in range(a.n)]
        used = set()
        chosen = []
        for w in want:
            for k in sorted(range(len(index)), key=lambda k: abs(index[k][1] - w)):
                if k not in used:
                    used.add(k)
                    chosen.append(k)
                    break
        require(len(chosen) == a.n, 'picked %d of %d seeds' % (len(chosen), a.n))
        chosen.sort(key=lambda k: index[k][0])
        sel = 'frame nearest each of %d equally spaced WPD-RMSD values (paper text)' % a.n
    cr = sorted((index[k][1] for k in chosen))
    gaps = [cr[i + 1] - cr[i] for i in range(len(cr) - 1)]
    even = (hi - lo) / (a.n - 1)
    print('  chosen: WPD RMSD %.3f - %.3f A (full range %.3f - %.3f), largest gap %.3f A vs %.3f A if perfectly even' % (cr[0], cr[-1], lo, hi, max(gaps), even))
    idxs = []
    for (n, k) in enumerate(chosen, start=1):
        (abs_step, rmsd, seg, frame) = index[k]
        pdb = out / ('snapshot_%d.pdb' % n)
        cpptraj(['parm %s' % (run / 'system.prmtop'), 'trajin %s %d %d 1' % (run / ('seg_%s.nc' % seg), frame, frame), 'strip %s' % STRIP, 'trajout %s pdb' % pdb], scratch, 'extract_%d' % n)
        require(pdb.exists() and pdb.stat().st_size > 0, 'extraction produced no snapshot_%d.pdb' % n)
        idxs.append((n, k, abs_step, rmsd, seg, frame))
        if n == 1 or n % 20 == 0 or n == len(chosen):
            print('    snapshot_%-3d seg_%s frame %-6d %9.3f ns  RMSD %.3f A' % (n, seg, frame, abs_step * DT_PS / 1000.0, rmsd))
    (out / 'snapshot_idxs.txt').write_text(''.join(('%d %d\n' % (n, k) for (n, k, _, _, _, _) in idxs)))
    (out / 'SEEDS.json').write_text(json.dumps(dict(run=str(run), state=prog.get('state'), direction=prog.get('direction'), mode=a.mode, selection=sel, n_seeds=a.n, frames_considered=len(index), gapless=gapless, allow_partial=bool(a.allow_partial), missing_segments=missing, worst_frame_offset_steps=drift, h3_done=(run / 'H3_DONE').exists(), ntwx=a.ntwx, stride_ns=stride_ns, last_frame_ns=last_ns, duration_ns=dur, reference=str(ref), fit_mask=FIT_MASK, wpd_mask=WPD_MASK, strip_mask=STRIP, wpd_rmsd_range_A=[lo, hi], chosen_wpd_rmsd_range_A=[cr[0], cr[-1]], largest_gap_A=max(gaps), even_gap_A=even, authors_code='sMD_analysis.ipynb cell 15: snapshot_idxs = np.linspace(0, last_frame, snapshot_number, dtype=int)', paper_text='100 snapshots were extracted from each sMD trajectory (200 total per model), equally sampling the WPD loop RMSD range', segments=[dict(segment=s['segment'], status=s.get('status'), start_step=s['start_step'], valid_to_step=s.get('valid_to_step') or s.get('end_step'), planned_end_step=s.get('planned_end_step'), limit_used=limit_sources.get(s['segment'])) for s in segs], chosen=[dict(snapshot=n, trajectory_frame=k, segment=seg, cpptraj_frame=frame, absolute_step=st, ns=round(st * DT_PS / 1000.0, 6), wpd_rmsd_to_closed_A=rm) for (n, k, st, rm, seg, frame) in idxs]), indent=2, ensure_ascii=False) + '\n')
    print('  wrote %d dry snapshots to %s' % (len(idxs), out))
if __name__ == '__main__':
    main()
