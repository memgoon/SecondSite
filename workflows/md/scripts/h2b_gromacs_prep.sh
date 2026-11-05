#!/bin/bash
set -eo pipefail
STATE=${1:?usage: h2b_gromacs_prep.sh <closed|open> [threads]}
NT=${2:-16}
R=${SECONDSITE_MD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PHOS=${PHOS:-$R/params}
SRC=$R/systems_author/2e__$STATE
OUT=$R/prepared_gmx/2e__$STATE
source "${GMXRC:-/opt/gromacs-2020.2/bin/GMXRC}"
export PATH=/opt/conda/bin:$PATH

for f in protein_substrate_heavy.pdb ligand.mol2 ligand.frcmod; do
  [ -s "$SRC/$f" ] || { echo "missing $SRC/$f (run h1b_author_input.py first)"; exit 1; }
done
mkdir -p "$OUT"; cd "$OUT"
cp "$SRC"/{protein_substrate_heavy.pdb,ligand.mol2,ligand.frcmod} .

leap_preamble() {
cat <<EOF
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
addAtomTypes {
{ "OP" "O" "sp3" } { "OQ" "O" "sp3" } { "OR" "O" "sp3" } { "OT" "O" "sp3" }
{ "OV" "O" "sp3" } { "OX" "O" "sp3" } { "OZ" "O" "sp3" }
}
loadamberparams $PHOS/frcmod.phosaa10
loadoff $PHOS/PTR.lib
loadoff $PHOS/SEP.lib
loadoff $PHOS/TPO.lib
loadoff $PHOS/Y1P.lib
loadoff $PHOS/S1P.lib
loadoff $PHOS/T1P.lib
loadamberparams ligand.frcmod
lig = loadmol2 ligand.mol2
u = loadpdb protein_substrate_heavy.pdb
u = combine { u lig }
EOF
}

echo "## 1/7 dry topology"
{ leap_preamble; echo "check u"; echo "saveamberparm u dry.prmtop dry.inpcrd"; echo quit; } > 00_dry.leap
tleap -f 00_dry.leap > 00_dry.log 2>&1
grep -qiE 'FATAL|does not have a type' 00_dry.log && { grep -iE 'FATAL|does not have a type' 00_dry.log | sort -u; exit 1; }
SOLUTE_Q=$(python -c "
import sys
t=[];k=None
for L in open('dry.prmtop'):
    if L.startswith('%FLAG'):k=L.split()[1];continue
    if L.startswith('%') or k!='CHARGE':continue
    t+=[float(x) for x in L.split()]
print(round(sum(t)/18.2223))")
echo "   solute net charge = $SOLUTE_Q e"

echo "## 2/7 solvate (TIP3P, 10 A shell) and size the salt"
{ leap_preamble; echo "solvatebox u TIP3PBOX 10.0"; echo "saveamberparm u solv.prmtop solv.inpcrd"; echo quit; } > 01_solvate.leap
tleap -f 01_solvate.leap > 01_solvate.log 2>&1
read NA NCL PAIRS VOL <<<"$(python -c "
import re
L=open('solv.inpcrd').read().splitlines()
b=[float(x) for x in L[-1].split()[:3]]
vol=b[0]*b[1]*b[2]
pairs=round(0.15*6.02214076e23*vol*1e-27)
q=$SOLUTE_Q
print(pairs+max(0,-q), pairs+max(0,q), pairs, round(vol,1))")"
echo "   box $VOL A^3 -> $PAIRS salt pairs ; addions2 Na+ $NA Cl- $NCL"
[ "$PAIRS" -gt 0 ] || { echo "salt pairs came out zero"; exit 1; }

echo "## 3/7 neutralise + 150 mM NaCl"
{ leap_preamble; echo "solvatebox u TIP3PBOX 10.0"; echo "addions2 u Na+ $NA Cl- $NCL"
  echo "saveamberparm u system.prmtop system.inpcrd"; echo "savepdb u system.pdb"; echo quit; } > 02_ions.leap
tleap -f 02_ions.leap > 02_ions.log 2>&1
grep -qiE 'FATAL' 02_ions.log && { grep -iE 'FATAL' 02_ions.log | sort -u; exit 1; }
python -c "
t=[];k=None
for L in open('system.prmtop'):
    if L.startswith('%FLAG'):k=L.split()[1];continue
    if L.startswith('%') or k!='CHARGE':continue
    t+=[float(x) for x in L.split()]
q=sum(t)/18.2223
assert abs(q)<1e-3, 'system not neutral: %.4f e'%q
print('   neutral: %+.4f e'%q)"

echo "## 4/7 Amber -> GROMACS (parmed)"
python - <<'PY'
import parmed as pmd
p=pmd.load_file('system.prmtop',xyz='system.inpcrd')
p.save('system.top',overwrite=True)
p.save('system.gro',overwrite=True)
print('   wrote system.top / system.gro  (%d atoms)'%len(p.atoms))
PY

echo "## 5/7 GROMACS minimisation, 7500 steepest descent steps"
gmx grompp -f $R/mdp/em.mdp   -c system.gro -p system.top -o em.tpr   -maxwarn 5 > grompp_em.log 2>&1
gmx mdrun  -deffnm em   -nt $NT -ntmpi 1 > mdrun_em.log 2>&1
echo "   $(grep -m1 -A2 'Steepest Descents converged\|reached the maximum number' em.log | head -2 | tr '\n' ' ')"

echo "## 6/7 heating 0 -> 300 K over 100 ps, then NPT 250 ps at 1 bar"
gmx grompp -f $R/mdp/heat.mdp -c em.gro     -p system.top -o heat.tpr -maxwarn 5 > grompp_heat.log 2>&1
gmx mdrun  -deffnm heat -nt $NT -ntmpi 1 > mdrun_heat.log 2>&1
gmx grompp -f $R/mdp/npt.mdp  -c heat.gro -t heat.cpt -p system.top -o npt.tpr -maxwarn 5 > grompp_npt.log 2>&1
gmx mdrun  -deffnm npt  -nt $NT -ntmpi 1 > mdrun_npt.log 2>&1
for f in em.gro heat.gro npt.gro; do [ -s $f ] || { echo "missing $f"; exit 1; }; done

echo "## 7/7 back to Amber restart for pmemd, then QC"
echo 0 | gmx trjconv -s npt.tpr -f npt.gro -o npt_whole.gro -pbc mol -ur compact > trjconv.log 2>&1
[ -s npt_whole.gro ] || { echo "trjconv produced no npt_whole.gro"; tail -5 trjconv.log; exit 1; }
python - <<'PY'
import numpy as np, parmed as pmd
p=pmd.load_file('system.prmtop')
g=pmd.load_file('npt_whole.gro')
p.coordinates=g.coordinates; p.box=g.box
v=getattr(g,'velocities',None)
assert v is not None, 'npt_whole.gro carries no velocities - check trjconv flags'
v=np.asarray(v)
assert v.shape==(len(p.atoms),3), 'velocity shape %s vs %d atoms'%(v.shape,len(p.atoms))
p.velocities=v
p.save('equil.rst7',overwrite=True,format='rst7')
p.save('equil.pdb',overwrite=True)
import math
lines=open('equil.rst7').read().splitlines()
need=2+2*math.ceil(len(p.atoms)/2)+1
assert len(lines)>=need, 'equil.rst7 has %d lines, needs %d for velocities'%(len(lines),need)
print('   wrote equil.rst7 (with velocities, %d lines) / equil.pdb'%len(lines))
PY
python $R/scripts/h2b_qc.py --dir . --state $STATE
echo "H2' DONE for 2e__$STATE -> $OUT"
