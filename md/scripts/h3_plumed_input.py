# SPDX-License-Identifier: MIT
import argparse, json, os, re, shutil, subprocess, tempfile
from pathlib import Path
ROOT = Path(os.environ.get('SECONDSITE_MD_ROOT', str(Path(__file__).resolve().parents[1])))
CACHE = Path(os.environ.get('AMMO_REFERENCE_DIR', str(Path(__file__).resolve().parents[1] / 'refs')))
TEMPLATE = {'open_to_closed': 'ammo_examples__example_ptp1b__.defaults__open_to_closed.dat', 'closed_to_open': 'ammo_examples__example_ptp1b__.defaults__closed_to_open.dat'}
REFPDB = {'open_to_closed': 'ammo_examples__example_ptp1b__inputs__closed_ref.pdb', 'closed_to_open': 'ammo_examples__example_ptp1b__inputs__open_ref.pdb'}

def _tool(envvar, name, fallback):
    return os.environ.get(envvar) or shutil.which(name) or fallback
CPPTRAJ = os.environ.get('CPPTRAJ_BIN') or shutil.which('cpptraj') or 'cpptraj'
PLUMED = os.environ.get('PLUMED_BIN') or shutil.which('plumed') or 'plumed'
REF_LAST_RESIDUE = 283

def require(x, m):
    if not x:
        raise ValueError(m)

def mask_atoms(prmtop, mask):
    r = subprocess.run([CPPTRAJ, '-p', str(prmtop), '--mask', mask], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines()[1:]:
        f = line.split()
        if len(f) > 3 and f[0].isdigit():
            out.append(int(f[0]))
    require(out, 'mask selected nothing: %s\n%s' % (mask, r.stdout[-500:] + r.stderr[-500:]))
    return out

def atom_table(prmtop):
    r = subprocess.run([CPPTRAJ, '-p', str(prmtop), '--mask', '*'], capture_output=True, text=True)
    tab = {}
    for line in r.stdout.splitlines()[1:]:
        f = line.split()
        if len(f) > 4 and f[0].isdigit():
            tab[int(f[0])] = (f[1], int(f[2]), f[3])
    require(tab, 'could not read the atom table')
    return tab

def write_reference(prmtop, ref_pdb, measured, out_path):
    tab = atom_table(prmtop)
    coords = {}
    for l in Path(ref_pdb).read_text().splitlines():
        if l[:6] in ('ATOM  ', 'HETATM'):
            coords[int(l[22:26]), l[12:16].strip()] = (l[30:38], l[38:46], l[46:54])
    require(coords, 'no coordinates parsed from ' + str(ref_pdb))
    meas = set(measured)
    rows = 0
    nmeas = 0
    with open(out_path, 'w') as fh:
        for num in sorted(tab):
            (name, resnum, resname) = tab[num]
            if resnum > REF_LAST_RESIDUE:
                continue
            key = (resnum, name)
            require(key in coords, 'the authors reference lacks %s%d.%s - numbering mismatch' % (resname, resnum, name))
            (x, y, z) = coords[key]
            occbeta = '  0.00  1.00' if num in meas else '  1.00  0.00'
            if num in meas:
                nmeas += 1
            el = name[0] if not name[0].isdigit() else name[1]
            fh.write('ATOM  %5d %-4s %3s  %4d    %8s%8s%8s%s          %2s\n' % (num, name[:4], resname[:3], resnum, x, y, z, occbeta, el))
            rows += 1
        fh.write('END\n')
    return (rows, nmeas)

def initial_cv_values(dat, structure, natoms):
    require(Path(PLUMED).exists(), 'plumed not found at ' + PLUMED)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        body = [l for l in dat.splitlines() if not l.strip().startswith(('MOVINGRESTRAINT', '...', 'PRINT', 'STEP')) and 'KAPPA' not in l and ('AT0' not in l) and ('ARG=rmsd' not in l)]
        probe = td / 'probe.dat'
        probe.write_text('\n'.join(body) + '\nPRINT ARG=rmsd,tyr,stacking,phe FILE=%s\n' % (td / 'COLVAR'))
        for f in Path(dat_dir).glob('reference_*.pdb'):
            shutil.copy(f, td / f.name)
        r = subprocess.run([PLUMED, 'driver', '--plumed', str(probe), '--mf_pdb', str(structure), '--natoms', str(natoms)], capture_output=True, text=True, cwd=str(td))
        col = td / 'COLVAR'
        if not col.exists():
            raise ValueError('plumed driver produced no COLVAR:\n' + r.stdout[-1500:] + r.stderr[-1500:])
        for line in col.read_text().splitlines():
            if line.startswith('#'):
                continue
            f = line.split()
            if len(f) >= 5:
                return [float(x) for x in f[1:5]]
    raise ValueError('COLVAR had no data row')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--direction', required=True, choices=sorted(TEMPLATE))
    ap.add_argument('--prmtop', required=True, help='topology of the system PLUMED will see')
    ap.add_argument('--structure', required=True, help='the STARTING structure, same atom numbering as --prmtop; used only to read off the initial CV values')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--plumed', default=None, help='plumed binary (default: PATH, then PLUMED_BIN)')
    ap.add_argument('--cpptraj', default=None, help='cpptraj binary (default: PATH, then CPPTRAJ_BIN)')
    ap.add_argument('--skip-initial', action='store_true', help='leave the literal "initial" in AT0/AT1 instead of asking PLUMED')
    a = ap.parse_args()
    global PLUMED, CPPTRAJ
    if a.plumed:
        PLUMED = a.plumed
    if a.cpptraj:
        CPPTRAJ = a.cpptraj
    tpl = CACHE / TEMPLATE[a.direction]
    require(tpl.exists(), 'template missing: ' + str(tpl))
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    global dat_dir
    dat_dir = od
    lines = tpl.read_text().splitlines()

    def ver(b, *args):
        try:
            r = subprocess.run([b] + list(args), capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr).strip().splitlines()[0][:120]
        except Exception as e:
            return 'unavailable: %s' % e
    report = {'direction': a.direction, 'template': str(tpl), 'prmtop': str(a.prmtop), 'tools': {'plumed': PLUMED, 'plumed_version': ver(PLUMED, 'info', '--version'), 'cpptraj': CPPTRAJ, 'cpptraj_version': ver(CPPTRAJ, '--version')}, 'substitutions': [], 'reference_files': []}
    out = []
    refcount = 0
    for line in lines:
        if 'REFERENCE=' in line:
            refcount += 1
            m = re.search('REFERENCE=(\\S+)', line)
            mask = m.group(1)
            measured = mask_atoms(a.prmtop, mask)
            rp = od / ('reference_%d.pdb' % refcount)
            refpdb = CACHE / REFPDB[a.direction]
            require(refpdb.exists(), 'authors reference missing: ' + str(refpdb))
            (rows, nmeas) = write_reference(a.prmtop, refpdb, measured, rp)
            require(nmeas == len(measured), 'only %d of %d measured atoms fell inside the reference span' % (nmeas, len(measured)))
            line = line.replace('REFERENCE=' + mask, 'REFERENCE=reference_%d.pdb' % refcount)
            report['reference_files'].append({'file': rp.name, 'mask': mask, 'reference_atoms_total': rows, 'measured_atoms': nmeas, 'measured_atom_numbers': measured, 'weighting': 'measured occupancy 0.00 / beta 1.00 ; aligned occupancy 1.00 / beta 0.00', 'coordinates_from': str(refpdb)})
            report['substitutions'].append({'field': 'REFERENCE', 'mask': mask, 'replacement': 'reference_%d.pdb' % refcount})
        elif 'ATOMS=' in line:
            m = re.search('ATOMS=(\\S+)', line)
            mask = m.group(1)
            nums = mask_atoms(a.prmtop, mask)
            line = line.replace('ATOMS=' + mask, 'ATOMS=' + ','.join((str(x) for x in nums)))
            tab = atom_table(a.prmtop)
            report['substitutions'].append({'field': 'ATOMS', 'mask': mask, 'atoms': nums, 'resolved': ['%s%d.%s' % (tab[n][2], tab[n][1], tab[n][0]) for n in nums]})
        out.append(line)
    dat = '\n'.join(out) + '\n'
    if not a.skip_initial:
        natoms = len(atom_table(a.prmtop))
        vals = initial_cv_values(dat, a.structure, natoms)
        report['initial_cv_values'] = dict(zip(['rmsd_nm', 'tyr_rad', 'stacking_nm', 'phe_nm'], vals))
        txt = ','.join(('%.6f' % v for v in vals))
        dat = re.sub('AT0=initial,initial,initial,initial', 'AT0=' + txt, dat)
        dat = re.sub('AT1=initial,initial,initial,initial', 'AT1=' + txt, dat)
        require('initial' not in dat, 'the literal "initial" is still present')
    (od / 'plumed.dat').write_text(dat)
    (od / 'H3_PLUMED_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print('wrote %s' % (od / 'plumed.dat'))
    for s in report['substitutions']:
        if s['field'] == 'ATOMS':
            print('   ATOMS %-22s -> %-18s %s' % (s['mask'], ','.join(map(str, s['atoms'])), ' '.join(s['resolved'])))
        else:
            print('   REFERENCE %-22s -> %s' % (s['mask'], s['replacement']))
    for r in report['reference_files']:
        print('   %s : %d reference atoms, %d measured (beta=1)' % (r['file'], r['reference_atoms_total'], r['measured_atoms']))
    if 'initial_cv_values' in report:
        print('   initial CVs from plumed driver: ' + '  '.join(('%s=%.4f' % (k, v) for (k, v) in report['initial_cv_values'].items())))
if __name__ == '__main__':
    main()
