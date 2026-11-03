#!/bin/bash
set -eo pipefail
CHECK_ONLY=0; NT_OPT=""; PARENT_OPT=""
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift;;
    --threads)    NT_OPT=${2:?--threads needs a value}; shift 2;;
    --parent)     PARENT_OPT=${2:?--parent needs a value}; shift 2;;
    --)           shift; while [ $# -gt 0 ]; do POS+=("$1"); shift; done;;
    -*)           echo "h5_prep: unknown option $1"; exit 2;;
    *)            POS+=("$1"); shift;;
  esac
done
SEED=${POS[0]:?usage: h5_seed_prep.sh <seed dir> <snapshot.pdb> <ligand.mol2> <ligand.frcmod> [threads] [parent_dry.prmtop] [--check-only] [--threads N] [--parent FILE]}
SNAP=${POS[1]:?snapshot pdb}
LMOL2=${POS[2]:?ligand.mol2}
LFRC=${POS[3]:?ligand.frcmod}
NT=${NT_OPT:-${POS[4]:-8}}
if [ -z "${R:-}" ]; then
  SELFDIR=$(cd "$(dirname "$0")" && pwd)
  if [ -d "${SECONDSITE_MD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/mdp" ]; then
    R=${SECONDSITE_MD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
  else
    R=$(dirname "$SELFDIR")
  fi
fi
if [ -n "${PARENT_OPT:-}" ]; then PARENT_DRY=$PARENT_OPT
elif [ -n "${POS[5]:-}" ]; then PARENT_DRY=${POS[5]}
elif [ -s "$R/params/parent_dry.prmtop" ]; then PARENT_DRY=$R/params/parent_dry.prmtop
else PARENT_DRY=$R/prepared_gmx/2e__open/dry.prmtop; fi
if [ -z "${PHOS:-}" ]; then
  if [ -s "$R/params/frcmod.phosaa10" ]; then PHOS=$R/params
  else PHOS=${PHOS:-$R/params}
  fi
fi
[ -s "$PHOS/frcmod.phosaa10" ] || { echo "h5_prep: phosaa10 parameters not found at $PHOS"; exit 1; }
for f in "$SNAP" "$LMOL2" "$LFRC" "$PARENT_DRY"; do
  [ -s "$f" ] || { echo "h5_prep: missing $f"; exit 1; }
done
abspath(){ echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
SNAP=$(abspath "$SNAP"); LMOL2=$(abspath "$LMOL2"); LFRC=$(abspath "$LFRC")
PARENT_DRY=$(abspath "$PARENT_DRY")
if [ "$CHECK_ONLY" = 1 ]; then
  [ -d "$SEED" ] || { echo "NOT_PREPARED: $SEED does not exist"; exit 3; }
  SEED=$(cd "$SEED" && pwd)
else
  mkdir -p "$SEED"; SEED=$(cd "$SEED" && pwd)
fi
for f in "$R/mdp/em.mdp" "$R/mdp/heat.mdp" "$R/mdp/npt.mdp"; do
  [ -s "$f" ] || { echo "h5_prep: missing $f"; exit 1; }
done
md5of(){ md5sum "$1" | cut -d' ' -f1; }
PREP_SIG="snap=$(md5of "$SNAP") mol2=$(md5of "$LMOL2") frcmod=$(md5of "$LFRC") parent=$(md5of "$PARENT_DRY") em=$(md5of "$R/mdp/em.mdp") heat=$(md5of "$R/mdp/heat.mdp") npt=$(md5of "$R/mdp/npt.mdp")"
if [ "$CHECK_ONLY" = 1 ]; then
  if [ ! -s "$SEED/PREP_DONE" ]; then
    echo "NOT_PREPARED: $SEED/PREP_DONE is missing or empty"; exit 3; fi
  WAS=$(grep -m1 '^signature=' "$SEED/PREP_DONE" 2>/dev/null | sed 's/^signature=//' || true)
  if [ -z "$WAS" ]; then
    echo "NOT_PREPARED: PREP_DONE has no signature= line"; exit 3; fi
  if [ "$WAS" != "$PREP_SIG" ]; then
    echo "NOT_PREPARED: prep signature does not match these inputs"
    echo "  recorded:  $WAS"
    echo "  requested: $PREP_SIG"; exit 3; fi
  for f in system.prmtop system_dry.prmtop seed_equil.rst7; do
    want=$(grep -m1 "^out_$f=" "$SEED/PREP_DONE" 2>/dev/null | sed "s/^out_$f=//" || true)
    if [ ! -s "$SEED/$f" ]; then echo "NOT_PREPARED: missing output $f"; exit 3; fi
    if [ -z "$want" ]; then echo "NOT_PREPARED: PREP_DONE records no checksum for $f"; exit 3; fi
    if [ "$(md5of "$SEED/$f")" != "$want" ]; then
      echo "NOT_PREPARED: $f changed since prep"; exit 3; fi
  done
  echo "prepared"; exit 0
fi

if [ -f "$SEED/PREP_DONE" ]; then
  WAS=$(grep -m1 '^signature=' "$SEED/PREP_DONE" 2>/dev/null | sed 's/^signature=//')
  if [ "$WAS" = "$PREP_SIG" ]; then
    BAD=""
    for f in system.prmtop system_dry.prmtop seed_equil.rst7; do
      want=$(grep -m1 "^out_$f=" "$SEED/PREP_DONE" 2>/dev/null | sed "s/^out_$f=//")
      if [ ! -s "$SEED/$f" ]; then BAD="$BAD $f(missing)"
      elif [ -z "$want" ]; then BAD="$BAD $f(no recorded checksum)"
      elif [ "$(md5of "$SEED/$f")" != "$want" ]; then BAD="$BAD $f(changed)"; fi
    done
    if [ -z "$BAD" ]; then
      echo "h5_prep: $SEED already prepared from these exact inputs, outputs verified"; exit 0
    fi
    echo "h5_prep: PREP_DONE matches but its outputs do not:$BAD - redoing"
    mv -f "$SEED/PREP_DONE" "$SEED/PREP_DONE.superseded.$(date -u +%Y%m%dT%H%M%SZ)"
  else
    echo "REFUSING: $SEED/PREP_DONE was produced from DIFFERENT inputs."
    echo "  recorded:  $WAS"
    echo "  requested: $PREP_SIG"
    echo "  Move the directory aside, or delete PREP_DONE if you mean to re-prepare."
    exit 3
  fi
fi
[ "$CHECK_ONLY" = 1 ] || cp "$LFRC" "$SEED/ligand.frcmod"
cd "$SEED"
[ -r "${GMXRC:-/opt/gromacs-2020.2/bin/GMXRC}" ] && source "${GMXRC:-/opt/gromacs-2020.2/bin/GMXRC}" || true
export PATH=/opt/conda/bin:$PATH

echo "## 1/6 split the dry snapshot; ligand coordinates into the original mol2"
python - "$SNAP" "$LMOL2" <<'PY'
import sys
snap,mol2=sys.argv[1],sys.argv[2]
prot=[];lig={}
for l in open(snap):
    if l.startswith('TER'): prot.append(l); continue
    if l[:6] not in ('ATOM  ','HETATM'): continue
    if l[17:20].strip()=='L2E': lig[l[12:16].strip()]=(l[30:38],l[38:46],l[46:54])
    else: prot.append(l)
assert lig, 'no L2E residue in '+snap
open('protein_substrate.pdb','w').write(''.join(prot)+'END\n')
L=open(mol2).read().splitlines(); i=L.index('@<TRIPOS>ATOM')
mol2_names=[]
for l in L[i+1:]:
    if l.startswith('@'): break
    q=l.split()
    if len(q)>=6: mol2_names.append(q[1])
miss_in_snap=[x for x in mol2_names if x not in lig]
extra_in_snap=[x for x in lig if x not in mol2_names]
assert not miss_in_snap and not extra_in_snap, (
    'ligand atom names do not match. In ligand.mol2 but not in the snapshot: %s. '
    'In the snapshot but not in ligand.mol2: %s'%(miss_in_snap,extra_in_snap))
assert len(mol2_names)==len(set(mol2_names)), 'ligand.mol2 has duplicate atom names'
out=L[:i+1]; n=0
for l in L[i+1:]:
    if l.startswith('@'): out.append(l); continue
    p=l.split()
    if len(p)>=6 and p[1] in lig:
        x,y,z=lig[p[1]]
        out.append('%7s %-8s %9.4f %9.4f %9.4f %-8s %5s %-8s %11s'%(
            p[0],p[1],float(x),float(y),float(z),p[5],p[6],p[7],p[8] if len(p)>8 else '0.0'))
        n+=1
    else: out.append(l)
assert n==len(mol2_names)==len(lig), (
    'transplanted %d but mol2 has %d and the snapshot has %d'%(n,len(mol2_names),len(lig)))
open('ligand.mol2','w').write('\n'.join(out)+'\n')
print('   %d protein/peptide lines; ligand names match exactly both ways, '
      '%d/%d coordinates transplanted'
      %(sum(1 for x in prot if x[:4]=='ATOM' or x[:6]=='HETATM'),n,len(mol2_names)))
PY

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
u = loadpdb protein_substrate.pdb
u = combine { u lig }
EOF
}

echo "## 2/6 dry topology"
{ leap_preamble; echo "check u"
  echo "saveamberparm u system_dry.prmtop system_dry.inpcrd"; echo quit; } > 00_dry.leap
tleap -f 00_dry.leap > 00_dry.log 2>&1
[ -s system_dry.prmtop ] || { echo "tleap failed"; grep -iE 'fatal|error' 00_dry.log | head; exit 1; }
SOLUTE_Q=$(python -c "
import parmed as pmd;print('%.4f'%sum(a.charge for a in pmd.load_file('system_dry.prmtop').atoms))")
echo "   solute net charge $SOLUTE_Q"

echo "## 2b/6 the rebuilt seed must be the SAME SYSTEM as its parent"
python - "$PARENT_DRY" <<'PY'
import sys
import parmed as pmd
ref=sys.argv[1]
sig=lambda p:[(a.name,a.residue.name,a.residue.idx,a.type,round(a.charge,6))
              for a in pmd.load_file(p).atoms]
a=sig('system_dry.prmtop'); b=sig(ref)
if len(a)!=len(b):
    raise SystemExit('   FAIL: rebuilt seed has %d atoms, parent %s has %d'%(len(a),ref,len(b)))
bad=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
if bad:
    i=bad[0]
    raise SystemExit('   FAIL: %d atom(s) differ from the parent; first at index %d: '
                     'seed %s vs parent %s'%(len(bad),i,a[i],b[i]))
pa=pmd.load_file('system_dry.prmtop'); pb=pmd.load_file(ref)
ra=[r.name for r in pa.residues]; rb=[r.name for r in pb.residues]
if ra!=rb: raise SystemExit('   FAIL: residue sequence differs from the parent')
qa=sum(x.charge for x in pa.atoms); qb=sum(x.charge for x in pb.atoms)
if abs(qa-qb)>1e-4:
    raise SystemExit('   FAIL: net charge %.4f vs parent %.4f'%(qa,qb))
def bondsig(parm):
    out=[]
    for b in parm.bonds:
        i,j=sorted((b.atom1.idx,b.atom2.idx))
        k=round(b.type.k,6) if b.type is not None else None
        r=round(b.type.req,6) if b.type is not None else None
        out.append((i,j,k,r))
    return sorted(out)
ba,bb=bondsig(pa),bondsig(pb)
if len(ba)!=len(bb):
    raise SystemExit('   FAIL: %d bonds vs parent %d'%(len(ba),len(bb)))
dif=[i for i,(x,y) in enumerate(zip(ba,bb)) if x!=y]
if dif:
    i=dif[0]
    raise SystemExit('   FAIL: %d bond(s) differ from the parent; first: seed %s vs parent %s'
                     %(len(dif),ba[i],bb[i]))
def angsig(parm):
    return sorted((tuple(sorted((x.atom1.idx,x.atom3.idx)))+(x.atom2.idx,
                   round(x.type.k,6) if x.type else None,
                   round(x.type.theteq,6) if x.type else None)) for x in parm.angles)
aa,ab=angsig(pa),angsig(pb)
if aa!=ab:
    n=sum(1 for x,y in zip(aa,ab) if x!=y)
    raise SystemExit('   FAIL: %d angle term(s) differ from the parent'%(n or abs(len(aa)-len(ab))))
print('   identical to the parent: %d atoms, %d residues, net charge %.4f'%(len(a),len(ra),qa))
print('   %d bonds and %d angles match atom-for-atom including force constants'
      %(len(ba),len(aa)))
print('   (names, residue names and indices, Amber atom types and charges all match)')
PY

echo "## 3/6 solvate (TIP3P, 10 A shell) and size the salt"
{ leap_preamble; echo "solvatebox u TIP3PBOX 10.0"
  echo "saveamberparm u solv.prmtop solv.inpcrd"; echo quit; } > 01_solvate.leap
tleap -f 01_solvate.leap > 01_solvate.log 2>&1
[ -s solv.inpcrd ] || { echo "solvation failed"; grep -iE 'fatal|error' 01_solvate.log | head; exit 1; }
read NA NCL VOL PAIRS <<<"$(python - "$SOLUTE_Q" <<'PY'
import sys
q=float(sys.argv[1])
L=open('solv.inpcrd').read().splitlines()
bx,by,bz=[float(x) for x in L[-1].split()[:3]]
vol=bx*by*bz                                    # A^3
pairs=int(round(0.150*6.02214076e23*vol/1e27))
na=pairs+(int(round(-q)) if q<0 else 0)
cl=pairs+(int(round(q)) if q>0 else 0)
print(na,cl,int(vol),pairs)
PY
)"
echo "   box $VOL A^3 -> $PAIRS salt pairs; addions2 Na+ $NA Cl- $NCL"

echo "## 4/6 neutralise + 150 mM NaCl"
{ leap_preamble; echo "solvatebox u TIP3PBOX 10.0"; echo "addions2 u Na+ $NA Cl- $NCL"
  echo "saveamberparm u system.prmtop system.inpcrd"; echo quit; } > 02_ions.leap
tleap -f 02_ions.leap > 02_ions.log 2>&1
[ -s system.prmtop ] || { echo "ion addition failed"; grep -iE 'fatal|error' 02_ions.log | head; exit 1; }
python -c "
import parmed as pmd
s=pmd.load_file('system.prmtop')
q=sum(a.charge for a in s.atoms)
assert abs(q)<1e-3, 'system is not neutral: %.4f'%q
print('   %d atoms, net charge %.2e'%(len(s.atoms),q))"

echo "## 5/6 Amber -> GROMACS, then 7500 min + 100 ps heat + 250 ps NPT"
python -c "
import parmed as pmd
s=pmd.load_file('system.prmtop','system.inpcrd')
s.save('system.top',overwrite=True); s.save('system.gro',overwrite=True)"
gmx grompp -f $R/mdp/em.mdp   -c system.gro -p system.top -o em.tpr   -maxwarn 5 > grompp_em.log 2>&1
gmx mdrun  -deffnm em   -nt $NT -ntmpi 1 > mdrun_em.log 2>&1
gmx grompp -f $R/mdp/heat.mdp -c em.gro      -p system.top -o heat.tpr -maxwarn 5 > grompp_heat.log 2>&1
gmx mdrun  -deffnm heat -nt $NT -ntmpi 1 > mdrun_heat.log 2>&1
gmx grompp -f $R/mdp/npt.mdp  -c heat.gro -t heat.cpt -p system.top -o npt.tpr -maxwarn 5 > grompp_npt.log 2>&1
gmx mdrun  -deffnm npt  -nt $NT -ntmpi 1 > mdrun_npt.log 2>&1
[ -s npt.gro ] || { echo "NPT produced no npt.gro"; tail -20 mdrun_npt.log; exit 1; }

echo "## 6/6 back to an Amber restart WITH velocities"
echo 0 | gmx trjconv -s npt.tpr -f npt.gro -o npt_whole.gro -pbc mol -ur compact > trjconv.log 2>&1
[ -s npt_whole.gro ] || { echo "trjconv produced no npt_whole.gro"; tail -5 trjconv.log; exit 1; }
python -c "
import numpy as np, parmed as pmd
g=pmd.load_file('npt_whole.gro')
p=pmd.load_file('system.prmtop')
p.coordinates=g.coordinates
v=getattr(g,'velocities',None)
assert v is not None, 'npt_whole.gro carries no velocities - check trjconv flags'
p.velocities=np.asarray(v)
p.box=g.box
p.save('seed_equil.rst7',overwrite=True)
n=len(p.atoms)
import math
lines=open('seed_equil.rst7').read().splitlines()
expect=2+2*math.ceil(n/2)+1
assert len(lines)==expect, 'rst7 has %d lines, expected %d (velocities missing?)'%(len(lines),expect)
print('   seed_equil.rst7: %d atoms, coordinates + velocities + box'%n)"
{ echo "signature=$PREP_SIG"
  for f in system.prmtop system_dry.prmtop seed_equil.rst7; do
    echo "out_$f=$(md5sum "$f" | cut -d' ' -f1)"
  done
  echo "prepared=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "snapshot=$SNAP"
  echo "atoms=$(python -c "import parmed as pmd;print(len(pmd.load_file('system.prmtop').atoms))")"
  echo "solute_charge=$SOLUTE_Q"; echo "salt_pairs=$PAIRS"; } > PREP_DONE
rm -f solv.prmtop solv.inpcrd em.trr heat.trr npt.trr
echo "## PREP_DONE"
