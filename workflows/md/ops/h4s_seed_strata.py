# SPDX-License-Identifier: MIT
import argparse, json, sys
from pathlib import Path
FRJ_SITE = {189, 190, 193, 194, 197, 201, 277, 278, 280, 281}
ORDER = ('FRJ_site_contact', 'other_surface_contact', 'no_protein_contact')

def stratum(c):
    if c['mim_pairs'] == 0:
        return 'no_protein_contact'
    return 'FRJ_site_contact' if c['mim_frj'] > 0 else 'other_surface_contact'

def span(v):
    import numpy as np
    if not v:
        return None
    a = np.sort(np.asarray(v, dtype=float))
    gaps = a[1:] - a[:-1] if len(a) > 1 else None
    return dict(n=len(a), min=float(a[0]), max=float(a[-1]), median=float(np.median(a)), span=float(a[-1] - a[0]), largest_gap=float(gaps.max()) if gaps is not None and len(gaps) else 0.0, gap_at=float(a[:-1][int(gaps.argmax())]) if gaps is not None and len(gaps) else None)

def bar(v, lo, hi, nb=12, width=40):
    import numpy as np
    if not v or hi <= lo:
        return []
    (h, edges) = np.histogram(np.asarray(v, dtype=float), bins=nb, range=(lo, hi))
    mx = max(1, h.max())
    out = []
    for (i, c) in enumerate(h):
        out.append('    %5.2f - %5.2f  %-*s %d' % (edges[i], edges[i + 1], width, '#' * int(round(width * c / mx)), c))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', action='append', required=True, help='label:SEEDS.json:contact_json:pose_json (pose_json optional, "-")')
    ap.add_argument('--reference', default='equilibrated', choices=('equilibrated', 'docked'), help='which pose reference the retained-contact count uses')
    ap.add_argument('--expect', type=int, default=100, help='seeds required per leg; the tally refuses a different count')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    legs = {}
    for spec in a.leg:
        parts = spec.split(':')
        if len(parts) < 3:
            sys.exit('--leg needs label:SEEDS.json:contact_json[:pose_json]')
        (label, sj, cj) = (parts[0], parts[1], parts[2])
        pj = parts[3] if len(parts) > 3 and parts[3] not in ('', '-') else None
        S = json.loads(Path(sj).read_text())
        C = json.loads(Path(cj).read_text())
        P = json.loads(Path(pj).read_text()) if pj else None
        prob = []
        for (src, obj, fields) in ((sj, S, ('state', 'run')), (cj, C, ('state', 'dir'))):
            for k in fields:
                if not obj.get(k):
                    prob.append('%s carries no %r field, so its provenance cannot be checked. Re-generate it with the current tools.' % (src, k))
        if S.get('state') and S['state'] != label:
            prob.append('%s says state=%r but this leg is labelled %r' % (sj, S['state'], label))
        if C.get('state') and C['state'] != label:
            prob.append('%s says state=%r but this leg is labelled %r' % (cj, C['state'], label))
        if S.get('run') and C.get('dir') and (S['run'] != C['dir']):
            prob.append('source run differs: SEEDS.json %r vs contact json %r' % (S['run'], C['dir']))
        if P is not None:
            pd = P.get('dir')
            if not pd:
                prob.append('%s carries no %r field, so its provenance cannot be checked.' % (pj, 'dir'))
            else:
                if S.get('run') and S['run'] != pd:
                    prob.append('source run differs: SEEDS.json %r vs pose json %r' % (S['run'], pd))
                if '/%s/' % label not in pd.rstrip('/') + '/':
                    prob.append('%s describes %r, which does not look like leg %r' % (pj, pd, label))
                if P.get('state') and P['state'] != label:
                    prob.append('%s says state=%r but this leg is labelled %r' % (pj, P['state'], label))
        if prob:
            sys.exit('PROVENANCE MISMATCH for leg %r - refusing to tally:\n  %s' % (label, '\n  '.join(prob)))
        raw_chosen = S.get('chosen', [])
        if not raw_chosen:
            sys.exit('%s has no chosen[]' % sj)
        seen = {}
        for c in raw_chosen:
            st_ = int(c['absolute_step'])
            seen.setdefault(st_, []).append(c.get('snapshot'))
        dups = {k: v for (k, v) in seen.items() if len(v) > 1}
        if dups:
            sys.exit('%s lists %d duplicated absolute_step(s), e.g. step %d used by snapshots %s. The selection is not 100 unique frames; refusing.' % (sj, len(dups), sorted(dups)[0], dups[sorted(dups)[0]]))
        chosen = {int(c['absolute_step']): c for c in raw_chosen}
        if len(chosen) != a.expect:
            sys.exit('%s holds %d unique selected seed(s), expected %d. Pass --expect to change what is required.' % (sj, len(chosen), a.expect))
        cby = {}
        cdup = []
        for f in C.get('frames', []):
            if not f.get('exact'):
                sys.exit('%s was not produced in exact-step mode, so its frames are not the selected seeds. Re-run h3x with --steps-json.' % cj)
            st_ = int(f['step'])
            if st_ in cby:
                cdup.append(st_)
            else:
                cby[st_] = f
        if cdup:
            sys.exit('%s holds %d duplicated step(s), e.g. %d. A dictionary would have kept one arbitrarily; refusing instead.' % (cj, len(cdup), sorted(set(cdup))[0]))
        miss = [st_ for st_ in chosen if st_ not in cby]
        if miss:
            sys.exit('%d selected seed(s) have no contact row in %s, e.g. step %d. The tally would be partial; refusing.' % (len(miss), cj, miss[0]))
        extra = [st_ for st_ in cby if st_ not in chosen]
        if extra:
            sys.exit('%s holds %d step(s) that were NOT selected, e.g. %d. It does not describe this seed set; refusing.' % (cj, len(extra), extra[0]))
        pby = {}
        pdup = []
        for f in P.get('frames', []) if P else []:
            st_ = int(f['step'])
            if st_ in pby:
                pdup.append(st_)
            else:
                pby[st_] = f
        if pdup:
            sys.exit('%s holds %d duplicated step(s), e.g. %d; refusing.' % (pj, len(pdup), sorted(set(pdup))[0]))
        pmiss = [st_ for st_ in chosen if st_ not in pby] if P else list(chosen)
        rows = []
        for (st, ch) in sorted(chosen.items()):
            c = cby[st]
            p = pby.get(st)
            kept = None
            if p and a.reference in p:
                kept = p[a.reference].get('kept_contacts')
            rows.append(dict(leg=label, seed=ch.get('snapshot'), step=st, ns=ch.get('ns'), wpd=ch.get('wpd_rmsd_to_closed_A'), stratum=stratum(c), pairs=c['mim_pairs'], frj=c['mim_frj'], mim_min=c['mim_min'], kept=kept, lig_rmsd=p[a.reference].get('lig_rmsd_A') if p and a.reference in p else None, raw_empty=bool(c['raw_pairs'] == 0 and c['mim_pairs'] > 0)))
        legs[label] = dict(rows=rows, pose=bool(P), pose_missing=pmiss, seeds_json=sj, contact_json=cj, pose_json=pj, traj_wpd=S.get('wpd_rmsd_range_A'), chosen_wpd=S.get('chosen_wpd_rmsd_range_A'))
        print('%s: %d seed(s) joined%s' % (label, len(rows), '' if P else '  (NO pose json: retained-contact counts unavailable)'))
    allrows = [r for L in legs.values() for r in L['rows']]
    print('\nretained-contact reference: %s' % a.reference)

    def report(title, rows, leg=None):
        print('\n=== %s   (n=%d)' % (title, len(rows)))
        print('  %-24s %5s   %-28s %s' % ('stratum', 'n', 'WPD RMSD to closed (A)', 'note'))
        for k in ORDER:
            sub = [r for r in rows if r['stratum'] == k]
            s = span([r['wpd'] for r in sub if r['wpd'] is not None])
            if not sub:
                continue
            print('  %-24s %5d   %s' % (k, len(sub), 'min %.3f  median %.3f  max %.3f  span %.3f' % (s['min'], s['median'], s['max'], s['span']) if s else 'no WPD values'))
            if k == 'FRJ_site_contact':
                kept = [r['kept'] for r in sub if r['kept'] is not None]
                full = [r for r in sub if r['kept'] is not None and r['kept'] >= 1.0 - 1e-09]
                if kept:
                    sf = span([r['wpd'] for r in full if r['wpd'] is not None])
                    print('  %-24s %5d   %s' % ('  of which fully_retained', len(full), 'min %.3f  median %.3f  max %.3f  span %.3f' % (sf['min'], sf['median'], sf['max'], sf['span']) if sf else '-'))
                    import numpy as np
                    kk = np.asarray(kept, dtype=float)
                    anyk = [r for r in sub if r['kept'] is not None and r['kept'] > 0]
                    sk = span([r['wpd'] for r in anyk if r['wpd'] is not None])
                    print('  %-24s %5d   %s' % ('  of which kept>0', len(anyk), 'min %.3f  median %.3f  max %.3f  span %.3f' % (sk['min'], sk['median'], sk['max'], sk['span']) if sk else '-'))
                    print('  %-24s         kept-contact fraction ACROSS THE WHOLE FRJ group: min %.2f median %.2f max %.2f' % ('', kk.min(), float(np.median(kk)), kk.max()))
                    print("  %-24s         (a seed can touch an FRJ-site residue and still retain none of the reference's own contacts - that is why 1 and 2 are counted separately)" % '')
                else:
                    print('  %-24s         kept-contact fraction unavailable (no pose json)' % '')
        ra = sum((1 for r in rows if r['raw_empty']))
        print('  %-24s %5d   frames the RAW metric would have called empty' % ('[raw artefact]', ra))
    for (label, L) in legs.items():
        report('leg %s' % label, L['rows'])
    if len(legs) > 1:
        report('BOTH LEGS COMBINED', allrows)
    print('\n=== coverage on the FRJ_site_contact subset only')
    print('  Distance to closed.pdb: SMALL = closed-like, LARGE = open-like.')
    for (label, L) in list(legs.items()) + ([('COMBINED', dict(rows=allrows, traj_wpd=None, chosen_wpd=None))] if len(legs) > 1 else []):
        rows = L['rows']
        pocket = [r['wpd'] for r in rows if r['stratum'] == 'FRJ_site_contact' and r['wpd'] is not None]
        allw = [r['wpd'] for r in rows if r['wpd'] is not None]
        sp = span(pocket)
        sa = span(allw)
        print('\n  %s' % label)
        if not sp or not sa:
            print('    no WPD values')
            continue
        print('    all selected seeds   %5d   %.3f - %.3f A  (span %.3f)' % (sa['n'], sa['min'], sa['max'], sa['span']))
        print('    pocket-contacting    %5d   %.3f - %.3f A  (span %.3f = %.0f%% of the above)' % (sp['n'], sp['min'], sp['max'], sp['span'], 100.0 * sp['span'] / sa['span'] if sa['span'] else float('nan')))
        if L.get('traj_wpd'):
            t = L['traj_wpd']
            print('    whole trajectory             %.3f - %.3f A  (from SEEDS.json)' % (t[0], t[1]))
            print('    pocket subset covers         %.0f%% of the trajectory span' % (100.0 * sp['span'] / (t[1] - t[0]) if t[1] > t[0] else float('nan')))
        print('    largest gap inside the pocket subset: %.3f A%s' % (sp['largest_gap'], ' (after %.3f A)' % sp['gap_at'] if sp['gap_at'] is not None else ''))
        print('    even spacing would be %.3f A over %d seeds' % (sp['span'] / max(1, sp['n'] - 1), sp['n']))
        print('    histogram of the pocket subset over the FULL selected range:')
        for ln in bar(pocket, sa['min'], sa['max']):
            print(ln)
        empty = [i for (i, ln) in enumerate(bar(pocket, sa['min'], sa['max'])) if ln.rstrip().endswith(' 0')]
        if empty:
            print('    %d of 12 bins are EMPTY in the pocket subset - stated as a fact, not a verdict.' % len(empty))
    print('\n=== completeness (required for this to count as a full-coverage QC)')
    bad = []
    for (label, L) in legs.items():
        rows = L['rows']
        nc = len(rows)
        npose = sum((1 for r in rows if r['kept'] is not None))
        nwpd = sum((1 for r in rows if r['wpd'] is not None and r['wpd'] == r['wpd']))
        line = '  %-8s contacts %3d/%d   pose %3d/%d   finite WPD %3d/%d' % (label, nc, a.expect, npose, a.expect, nwpd, a.expect)
        ok = nc == a.expect and npose == a.expect and (nwpd == a.expect)
        print(line + ('   OK' if ok else '   INCOMPLETE'))
        if not ok:
            if nc != a.expect:
                bad.append('%s: %d contact row(s), expected %d' % (label, nc, a.expect))
            if npose != a.expect:
                mi = L.get('pose_missing') or []
                bad.append('%s: %d pose row(s), expected %d%s' % (label, npose, a.expect, ' (no pose json given)' if not L['pose'] else '; missing e.g. step %s' % (mi[0] if mi else '?')))
            if nwpd != a.expect:
                bad.append('%s: %d finite WPD value(s), expected %d' % (label, nwpd, a.expect))
    if bad:
        print('\n  THIS IS NOT A COMPLETED QC. Outstanding:')
        for x in bad:
            print('    - %s' % x)
        print('  The strata above are kept as evidence; do not read them as full coverage.')
    else:
        print('  both legs complete: %d contacts, %d pose rows and %d finite WPD values each' % (a.expect, a.expect, a.expect))
    print("\nNote: fully_retained means every residue the REFERENCE contacts is still contacted. It does NOT mean the original pose is preserved - the pose report's RMSD and centroid distributions are the place to look for that.")
    print('No verdict is given. Whether this coverage suffices for the MSM must be decided against a criterion written down beforehand.')
    if a.json:
        out = dict(reference=a.reference, legs={k: dict(seeds_json=v['seeds_json'], contact_json=v['contact_json'], pose_json=v['pose_json'], traj_wpd=v.get('traj_wpd'), rows=v['rows']) for (k, v) in legs.items()})
        for (k, v) in legs.items():
            out['legs'][k]['strata'] = {s: span([r['wpd'] for r in v['rows'] if r['stratum'] == s and r['wpd'] is not None]) for s in ORDER}
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
        print('\nrecord -> %s' % a.json)
    return 1 if bad else 0
if __name__ == '__main__':
    sys.exit(main())
