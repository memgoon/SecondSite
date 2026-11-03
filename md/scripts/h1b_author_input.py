# SPDX-License-Identifier: MIT
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('SECONDSITE_MD_ROOT', str(Path(__file__).resolve().parents[1])))
CACHE = Path(os.environ.get('AMMO_REFERENCE_DIR', str(Path(__file__).resolve().parents[1] / 'refs')))
OURS = Path(os.environ.get('CANDIDATE_INPUT_DIR', str(Path(__file__).resolve().parents[1] / 'candidate_inputs')))
AUTHOR = {'closed': 'ammo_examples__example_ptp1b__inputs__closed-peptide-1.pdb', 'open': 'ammo_examples__example_ptp1b__inputs__open-peptide-1.pdb'}
LIG_RES = 296
PROTONATION = {98: 'GLH', 182: 'ASH', 215: 'HID', 216: 'CYM'}
CORE = [(r + 1, 'CA') for r in range(5, 176)]

def require(x, m):
    if not x:
        raise ValueError(m)

def rd(p):
    out = []
    for l in open(p):
        if l[:6] not in ('ATOM  ', 'HETATM'):
            continue
        rn = l[17:20].strip()
        if rn in ('WAT', 'HOH', 'Na+', 'Cl-'):
            continue
        out.append(dict(name=l[12:16], res=rn, ri=int(l[22:26]), xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), el=l[76:78].strip() or l[12:16].strip()[:1]))
    return out

def heavy(a):
    return a['el'] != 'H' and (not a['name'].strip().startswith('H'))

def kabsch(m, t):
    (cm, ct) = (m.mean(0), t.mean(0))
    (u, s, v) = np.linalg.svd((m - cm).T @ (t - ct))
    rot = u @ np.diag([1, 1, np.linalg.det(u @ v)]) @ v
    return lambda p: (p - cm) @ rot + ct

def fmt(n, name, res, ri, xyz, el):
    return 'ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s' % (n, name, res, ri, xyz[0], xyz[1], xyz[2], el)

def build(state):
    ap = CACHE / AUTHOR[state]
    require(ap.exists(), 'author input missing: ' + str(ap))
    A = rd(ap)
    amap = {}
    for a in A:
        amap.setdefault(a['ri'], a['res'])
    require(len(amap) == 296, 'author file has %d residues' % len(amap))
    require(amap[LIG_RES] == 'FRJ', 'residue %d is %s, expected FRJ' % (LIG_RES, amap[LIG_RES]))
    for (r, n) in PROTONATION.items():
        require(amap[r] == n, 'author protonation at %d is %s, expected %s' % (r, amap[r], n))
    op = OURS / ('2e_C1_A__%s' % state) / 'protein_substrate_heavy.pdb'
    require(op.exists(), 'our 2e structure missing: ' + str(op))
    B = rd(op)
    bidx = {(b['ri'], b['name'].strip()): b['xyz'] for b in B}
    aidx = {(a['ri'], a['name'].strip()): a['xyz'] for a in A}
    shared = [k for k in CORE if k in aidx and k in bidx]
    require(len(shared) > 150, 'only %d shared core CA atoms' % len(shared))
    tf = kabsch(np.array([bidx[k] for k in shared]), np.array([aidx[k] for k in shared]))
    fit = float(np.sqrt(((tf(np.array([bidx[k] for k in shared])) - np.array([aidx[k] for k in shared])) ** 2).sum(1).mean()))
    ml = (OURS / ('2e_C1_A__%s' % state) / 'ligand.mol2').read_text().splitlines()
    i = ml.index('@<TRIPOS>ATOM')
    j = next((k for k in range(i + 1, len(ml)) if ml[k].startswith('@<TRIPOS>')))
    lig = []
    for row in ml[i + 1:j]:
        f = row.split()
        lig.append((f[1], np.array([float(f[2]), float(f[3]), float(f[4])]), 'H' if f[5].lower().startswith('h') else f[5][:1].upper()))
    nheavy = sum((1 for (nm, _, el) in lig if el != 'H'))
    require(nheavy == 16, '2e should have 16 heavy atoms, mol2 has %d (total %d)' % (nheavy, len(lig)))
    moved = tf(np.array([x[1] for x in lig]))
    heavy_mask = [k for (k, (nm, _, el)) in enumerate(lig) if el != 'H']
    lines = []
    n = 0
    kept = 0
    renamed = 0
    for a in A:
        if a['ri'] >= LIG_RES or not heavy(a):
            continue
        nm = a['name']
        if a['res'] == 'NME' and nm.strip() == 'CH3':
            nm = '   C'
            renamed += 1
        n += 1
        kept += 1
        lines.append(fmt(n, nm, a['res'], a['ri'], a['xyz'], a['el']))
    lines.append('TER')
    od = ROOT / 'systems_author' / ('2e__%s' % state)
    od.mkdir(parents=True, exist_ok=True)
    (od / 'protein_substrate_heavy.pdb').write_text('\n'.join(lines) + '\nEND\n')
    body = []
    for (row, xyz) in zip(ml[i + 1:j], moved):
        f = row.split()
        (f[2], f[3], f[4]) = ('%.4f' % xyz[0], '%.4f' % xyz[1], '%.4f' % xyz[2])
        body.append('%7s %-8s %9s %9s %9s %-6s %4s %-8s %9s' % (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]))
    (od / 'ligand.mol2').write_text('\n'.join(ml[:i + 1] + body + ml[j:]) + '\n')
    for f in ('ligand.frcmod',):
        (od / f).write_bytes((OURS / ('2e_C1_A__%s' % state) / f).read_bytes())
    frj = np.array([a['xyz'] for a in A if a['ri'] == LIG_RES and heavy(a)])
    prot = [(a['ri'], a['res'], a['xyz']) for a in A if a['ri'] < LIG_RES and heavy(a)]

    def site(L):
        out = {}
        for (ri, rn, p) in prot:
            d = float(np.min(np.linalg.norm(L - p, axis=1)))
            if d <= 4.5:
                out[ri] = (rn, d)
        return out
    moved_heavy = moved[heavy_mask]
    (s_frj, s_2e) = (site(frj), site(moved_heavy))
    mind = float(np.min([np.min(np.linalg.norm(moved_heavy - p, axis=1)) for (_, _, p) in prot]))
    rep = dict(state=state, source=str(ap), author_atoms=len(A), kept_solute_heavy=kept, protonation_verified=PROTONATION, core_fit_atoms=len(shared), core_fit_RMSD_A=fit, ligand_atoms_total=len(lig), ligand_heavy_atoms=nheavy, frj_centroid_to_2e_centroid_A=float(np.linalg.norm(frj.mean(0) - moved_heavy.mean(0))), frj_site_residues=sorted(s_frj), contact_residues_2e=sorted(s_2e), shared_site_residues=sorted(set(s_frj) & set(s_2e)), site_overlap_fraction_of_2e=round(len(set(s_frj) & set(s_2e)) / max(1, len(s_2e)), 3), min_2e_to_protein_heavy_A=mind, nme_CH3_renamed_to_C=renamed, nme_rename_reason='ff14SB aminoct12.lib defines NME methyl carbon as "C"; authors PDB uses "CH3"', hydrogens='dropped, rebuilt by tleap as in the authors BioSimSpace setup')
    print('%-7s core fit %.3f A on %d CA | 2e-FRJ centroid %.2f A | 2e contacts %d, %d shared with FRJ site (%.0f%%) | min 2e-protein %.2f A' % (state, fit, len(shared), rep['frj_centroid_to_2e_centroid_A'], len(s_2e), len(rep['shared_site_residues']), 100 * rep['site_overlap_fraction_of_2e'], mind))
    return rep

def main():
    reps = [build(s) for s in ('closed', 'open')]
    (ROOT / 'H1B_REPORT.json').write_text(json.dumps(dict(approach='authors own compound-1 input structures with FRJ replaced by 2e', author_files=AUTHOR, numbering='author residue = PTP1B residue + 1; identical to ours', supersedes='h1_truncate.py / systems/ (built from this project lineage)', states=reps), indent=2) + '\n')
    print('\nwrote', ROOT / 'H1B_REPORT.json')
if __name__ == '__main__':
    main()
