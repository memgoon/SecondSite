#!/bin/bash
set -eo pipefail
STATE=""; PREP=""; OUT=""; NSTLIM=76000000; TAG=""; PLDIR=""; RESUME=0; FORCE=0
while [ $# -gt 0 ]; do case "$1" in
  --state)  STATE=$2; shift 2;;
  --prep)   PREP=$2;  shift 2;;
  --out)    OUT=$2;   shift 2;;
  --nstlim) NSTLIM=$2; shift 2;;
  --tag)    TAG=$2;   shift 2;;
  --plumed-dir) PLDIR=$2; shift 2;;
  --resume) RESUME=1; shift;;
  --force)  FORCE=1;  shift;;
  *) echo "unknown argument: $1"; exit 2;;
esac; done
[ -n "$STATE" ] && [ -n "$PREP" ] && [ -n "$OUT" ] || {
  echo "usage: --state <closed|open> --prep <dir> --out <dir> [--plumed-dir d]"
  echo "       [--nstlim N] [--tag name] [--resume] [--force]"; exit 2; }
case "$STATE" in
  open)   DIR=open_to_closed;;
  closed) DIR=closed_to_open;;
  *) echo "state must be closed or open"; exit 2;;
esac
DT=0.002
R=${SECONDSITE_MD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
SELF=$(cd "$(dirname "$0")" && pwd)
OFFSET_PY=$SELF/h3c_plumed_offset.py; [ -s "$OFFSET_PY" ] || OFFSET_PY=$R/scripts/h3c_plumed_offset.py
[ -s "$OFFSET_PY" ] || { echo "h3c_plumed_offset.py not found"; exit 1; }

if [ -f "$OUT/H3_DONE" ] && [ "$FORCE" = 0 ]; then
  echo "REFUSING: $OUT already holds a completed run (H3_DONE)."
  echo "  nothing to resume; use --force only to deliberately discard it."; exit 3
fi
if [ "$RESUME" = 0 ] && [ "$FORCE" = 0 ] && [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "REFUSING: $OUT is not empty and this is not a resume."
  echo "  use --resume to continue, --force to overwrite, or pick another --out."; exit 3
fi
for f in system.prmtop equil.rst7 equil.pdb; do
  [ -s "$PREP/$f" ] || { echo "missing $PREP/$f (run h2b_gromacs_prep.sh first)"; exit 1; }
done
mkdir -p "$OUT"; cd "$OUT"

[ -r "${GMXRC:-/opt/gromacs-2020.2/bin/GMXRC}" ] && source "${GMXRC:-/opt/gromacs-2020.2/bin/GMXRC}" || true
export PATH=/opt/pmemd26/bin:/opt/plumed-2.6.1/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/plumed-2.6.1/lib:${LD_LIBRARY_PATH:-}
export PLUMED_KERNEL=${PLUMED_KERNEL:-/opt/plumed-2.6.1/lib/libplumedKernel.so}

rst_time(){ # ps recorded inside an Amber restart (line 2 is "natom time")
  awk 'NR==2{ if (NF>=2) printf "%.6f", $2; else print "NONE"; exit }' "$1"; }

if [ "$RESUME" = 0 ]; then
  cp "$PREP"/system.prmtop "$PREP"/equil.rst7 "$PREP"/equil.pdb .
  if [ -n "$PLDIR" ]; then
    echo "## 1/5 pre-generated PLUMED input from $PLDIR"
    cp "$PLDIR"/plumed.dat "$PLDIR"/reference_*.pdb .
    cp "$PLDIR"/H3_PLUMED_REPORT.json . 2>/dev/null || true
  else
    echo "## 1/5 generate the PLUMED input for $DIR (needs the AMMo cache)"
    python $R/scripts/h3_plumed_input.py --direction "$DIR" \
      --prmtop system.prmtop --structure equil.pdb --outdir . | sed 's/^/   /'
  fi
  [ -s plumed.dat ] && [ -s reference_1.pdb ] || { echo "plumed input missing"; exit 1; }
  cp plumed.dat plumed_original.dat
  SEG=0; DONE=0; START_C=equil.rst7; NTX=1; IREST=0
else
  echo "## 1/5 RESUME  (a DEVIATION: the authors run the 152 ns in one piece, so"
  echo "              MOVINGRESTRAINT's accumulated work restarts at each leg)"
  [ -s plumed_original.dat ] || { echo "plumed_original.dat missing"; exit 1; }
  LAST=$(ls seg_*.rst 2>/dev/null | sort | tail -1 || true)
  [ -n "$LAST" ] && [ -s "$LAST" ] || { echo "no seg_*.rst to resume from"; exit 1; }
  RST_PS=$(rst_time "$LAST")
  [ "$RST_PS" != NONE ] || { echo "$LAST carries no time stamp; cannot resume safely"; exit 1; }
  DONE=$(python -c "print(int(round($RST_PS/$DT)))")
  SEG=$(( $(echo "$LAST" | sed 's/seg_0*\([0-9]*\)\.rst/\1/') + 1 ))
  echo "   restart $LAST records ${RST_PS} ps -> ${DONE} steps done; next leg seg_$(printf %03d $SEG)"
  python - "$LAST" "$DONE" <<'PY'
import json,os,re,sys
last,done=sys.argv[1],int(sys.argv[2])
s3=re.sub(r'seg_0*([0-9]+)\.rst',r'\1',last).zfill(3)
p='PROGRESS.json'
d=json.load(open(p)) if os.path.exists(p) else {'segments':[]}
prev=next((x for x in d['segments'] if x['segment']==s3),None)
if prev is None:
    earlier=[x for x in d['segments'] if x['segment']<s3 and 'end_step' in x]
    start=max((x['end_step'] for x in earlier),default=0)
    if start==0 and s3!='000':
        raise SystemExit('   FAIL: seg_%s is unregistered and no earlier segment '
                         'records an end_step, so its start step is unknown. '
                         'Refusing to guess 0.'%s3)
    prev=dict(segment=s3,start_step=start,files=dict(out='seg_%s.out'%s3,rst=last,
              nc='seg_%s.nc'%s3,mdin='seg_%s.mdin'%s3,plumed='seg_%s.plumed.dat'%s3))
    d['segments'].append(prev)
    print('   recovered seg_%s start_step=%d from the previous segment'%(s3,start))
prev.update(status='partial',valid_to_step=done,
            end_step=done,steps=done-prev.get('start_step',0),
            note='leg did not finish; trajectory is valid up to valid_to_step')
d['segments'].sort(key=lambda x:x['segment'])
d['steps_done']=max(x.get('end_step',0) for x in d['segments'])
d['ns_done']=round(d['steps_done']*0.002/1000,4)
json.dump(d,open(p,'w'),indent=2)
print('   registered seg_%s as PARTIAL, valid to step %d'%(s3,done))
PY
  cp "$LAST" "resume_from_$(printf %03d $SEG).rst7"
  START_C="resume_from_$(printf %03d $SEG).rst7"; NTX=5; IREST=1
  NSTLIM=$(( NSTLIM - DONE ))
  [ "$NSTLIM" -gt 0 ] || { echo "already at or past the requested step count"; exit 0; }
fi
S3=$(printf %03d $SEG)
CV="seg_$S3.colvar"
python "$OFFSET_PY" --plumed plumed_original.dat --offset "$DONE" \
  --colvar "$CV" --out plumed.dat | sed 's/^/   /'
echo "   schedule in effect for seg_$S3 (PLUMED clock restarts at 0 each leg):"
grep -E 'STEP[0-9]|PRINT' plumed.dat | sed 's/^/     /'
cp plumed.dat "seg_$S3.plumed.dat"

echo "## 2/5 mdin for seg_$S3 (ntx=$NTX irest=$IREST)"
cat > "seg_$S3.mdin" <<EOF
Production.
 &cntrl
  ig=-1,
  ntx=$NTX,
  ntxo=1,
  ntpr=2500,
  ntwr=2500,
  ntwx=2500,
  irest=$IREST,
  dt=$DT,
  nstlim=$NSTLIM,
  ntc=2,
  ntf=2,
  ntt=3,
  gamma_ln=2,
  cut=8.0,
  tempi=300.00,
  temp0=300.00,
  ntp=1,
  pres0=1.01325,
  plumed=1,
  plumedfile='plumed.dat',
 /
EOF
sed 's/^/   /' "seg_$S3.mdin"

python - "$S3" "$DONE" "$NSTLIM" "$START_C" "$STATE" "$DIR" <<'PY'
import json,os,sys
s3,done,nst,startc,state,direction=sys.argv[1:7]
p='PROGRESS.json'
d=json.load(open(p)) if os.path.exists(p) else {'segments':[]}
d.update(state=state,direction=direction)
d['segments']=[x for x in d['segments'] if x['segment']!=s3]
d['segments'].append(dict(segment=s3,start_step=int(done),status='running',
    planned_steps=int(nst),planned_end_step=int(done)+int(nst),input_restart=startc,
    files=dict(mdin='seg_%s.mdin'%s3,plumed='seg_%s.plumed.dat'%s3,
               out='seg_%s.out'%s3,rst='seg_%s.rst'%s3,nc='seg_%s.nc'%s3,
               colvar='seg_%s.colvar'%s3)))
d['segments'].sort(key=lambda x:x['segment'])
json.dump(d,open(p,'w'),indent=2)
print('   pre-registered seg_%s as running, start_step=%d'%(s3,int(done)))
PY

echo "## 3/5 pmemd.cuda  seg_$S3: $NSTLIM steps = $(python -c "print('%.3f'%($NSTLIM*$DT/1000))") ns"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader | sed 's/^/   /'
T0=$(date +%s)
pmemd.cuda -O -i "seg_$S3.mdin" -p system.prmtop -c "$START_C" \
  -o "seg_$S3.out" -r "seg_$S3.rst" -x "seg_$S3.nc" -inf "seg_$S3.info" 2>&1 | tail -5 | sed 's/^/   /'
T1=$(date +%s)

echo "## 4/5 verify seg_$S3"
for f in "seg_$S3.rst" "seg_$S3.out" "$CV"; do
  [ -s "$f" ] || { echo "FAIL: $f missing or empty"; tail -30 "seg_$S3.out" 2>/dev/null; exit 1; }
done
grep -qi plumed "seg_$S3.out" || { echo "FAIL: seg_$S3.out never mentions PLUMED"; exit 1; }
FIELDS=$(grep -m1 '^#! FIELDS' "$CV" | sed 's/^#! FIELDS //')
echo "   COLVAR fields: $FIELDS"
col(){ awk -v want="$1" -v mode="$2" '
  /^#! FIELDS/ { for(i=3;i<=NF;i++) if($i==want) c=i-2; next }
  /^#/ { next } { if(c){ v=$c+0; if(f=="")f=v; l=v; if(mx==""||v>mx)mx=v } }
  END{ if(!c) exit 2; if(mode=="first")print f; else if(mode=="last")print l; else print mx }' "$CV"; }
for cv in rmsd tyr stacking phe; do
  a=$(col $cv first) || { echo "FAIL: required column '$cv' absent from COLVAR"; exit 1; }
  printf "   %-9s %12s -> %12s\n" "$cv" "$a" "$(col $cv last)"
done
BIAS=$(echo "$FIELDS" | tr ' ' '\n' | grep -m1 '\.bias$' || true)
[ -n "$BIAS" ] || { echo "FAIL: no *.bias column in COLVAR - the restraint is not reported"; exit 1; }
BMAX=$(col "$BIAS" max); BLAST=$(col "$BIAS" last)
echo "   $BIAS  max=$BMAX final=$BLAST"
KAPPA_COLS=$(echo "$FIELDS" | tr ' ' '\n' | grep '_kappa$' || true)
CNTR_COLS=$(echo "$FIELDS" | tr ' ' '\n' | grep '_cntr$'  || true)
[ -n "$KAPPA_COLS" ] || { echo "FAIL: no *_kappa columns in COLVAR"; exit 1; }
[ -n "$CNTR_COLS" ]  || { echo "FAIL: no *_cntr columns in COLVAR"; exit 1; }
NK=$(echo "$KAPPA_COLS" | wc -l); NC=$(echo "$CNTR_COLS" | wc -l)
[ "$NK" -eq 4 ] && [ "$NC" -eq 4 ] || {
  echo "FAIL: expected 4 *_kappa and 4 *_cntr columns (one per CV), got $NK and $NC"; exit 1; }
for k in $KAPPA_COLS; do
  printf "   %-22s max=%-12s final=%s\n" "$k" "$(col "$k" max)" "$(col "$k" last)"
done
awk -v x="$BMAX" 'BEGIN{exit !(x>0)}' \
  || { echo "FAIL: bias never became non-zero - the restraint never acted"; exit 1; }
RST_PS=$(rst_time "seg_$S3.rst"); CPS=$(grep -v '^#' "$CV" | tail -1 | awk '{print $1}')
EXP_PS=$(python -c "print('%.6f'%(($DONE+$NSTLIM)*$DT))")
AMBER_STEP=$(grep -oE 'NSTEP =[ ]*[0-9]+' "seg_$S3.out" | tail -1 | grep -oE '[0-9]+$' || echo "")
LEG_PS=$(python -c "print('%.6f'%($NSTLIM*$DT))")
echo "   progress: restart=${RST_PS} ps (absolute)  $CV=${CPS} ps (local, leg is ${LEG_PS} ps)"
echo "             amber NSTEP=${AMBER_STEP:-n/a}  expected absolute=${EXP_PS} ps"
python -c "
import sys
dt=$DT; stride_ps=2500*dt
r=float('$RST_PS'); c=float('$CPS'); e=float('$EXP_PS'); leg=float('$LEG_PS')
amber='$AMBER_STEP'
bad=[]
if abs(r-e)>2*dt: bad.append('restart %.6f vs expected absolute %.6f ps (tol %.4f)'%(r,e,2*dt))
if not (leg-stride_ps-dt <= c <= leg+dt):
    bad.append('%s %.6f outside [%.6f, %.6f] ps local (one PRINT stride)'
               %('$CV',c,leg-stride_ps-dt,leg+dt))
if amber and abs(int(amber)-$NSTLIM)>2:
    bad.append('amber NSTEP %s is not this leg\'s %d steps'%(amber,$NSTLIM))
if bad: print('FAIL: '+'; '.join(bad)); sys.exit(1)
" || exit 1
TOTAL=$(( DONE + NSTLIM ))
echo "   final KAPPA against the original schedule at the last reported step:"
python - "$CV" "$DONE" "$DT" "$SELF" "$R" "$TOTAL" <<'PY' || exit 1
import os,sys,importlib.util
cv=sys.argv[1];done=int(sys.argv[2]);dt=float(sys.argv[3])
self_dir,repo,total=sys.argv[4],sys.argv[5],int(sys.argv[6])
h3d=None
for c in (os.path.join(self_dir,'h3d_verify_resume.py'),
          os.path.join(repo,'scripts','h3d_verify_resume.py')):
    if os.path.exists(c):h3d=c;break
if h3d is None:
    print('   FAIL: h3d_verify_resume.py not found; cannot evaluate the schedule')
    sys.exit(1)
spec=importlib.util.spec_from_file_location('h3dmod',h3d)
m=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
spec.loader.exec_module(m)
args,steps=m.parse_schedule('plumed_original.dat')
fields,rows=m.read_colvar(cv)
if not rows:
    print('   FAIL: %s holds no data rows'%cv);sys.exit(1)
idx={f:i for i,f in enumerate(fields)}
last=rows[-1]
abs_step=done+int(round(last[0]/dt))
_,ekp=m.at_step(steps,abs_step)
bad=[]
for j,a in enumerate(args):
    hits=[f for f in fields if f.endswith('.%s_kappa'%a)]
    if len(hits)!=1:
        print('   FAIL: expected one *.%s_kappa column, found %s'%(a,hits));sys.exit(1)
    got=last[idx[hits[0]]];exp=ekp[j]
    ok=abs(got-exp)<=max(1.0,0.001*max(abs(exp),1.0))
    print('     %-10s reported %10.4f   schedule %10.4f   %s'
          %(a,got,exp,'ok' if ok else 'MISMATCH'))
    if not ok:bad.append('%s %.4f vs %.4f'%(a,got,exp))
print('     last reported row: local %.3f ps -> absolute step %d'%(last[0],abs_step))
if bad:
    print('   FAIL: final KAPPA does not match the authors schedule: '+'; '.join(bad))
    sys.exit(1)
if total>=76000000:
    s2=steps[-2][0]
    if abs_step<s2:
        print('   FAIL: the run claims %d steps but the last reported row is step '
              '%d, before STEP2=%d, so the decay ramp was never reached'
              %(total,abs_step,s2))
        sys.exit(1)
    print('     decay ramp entered (last row step %d >= STEP2 %d)'%(abs_step,s2))
PY

echo "## 5/5 record and stitch the absolute timeline"
python - "$S3" "$DONE" "$NSTLIM" "$RST_PS" "$STATE" "$DIR" "$RESUME" "$((T1-T0))" <<'PY'
import json,os,sys
s3,done,nst,rst,state,direction,res,wall=sys.argv[1:9]
p='PROGRESS.json'
d=json.load(open(p)) if os.path.exists(p) else {'segments':[]}
d.update(state=state,direction=direction)
d['segments']=[x for x in d['segments'] if x['segment']!=s3]
d['segments'].append(dict(segment=s3,start_step=int(done),steps=int(nst),
    end_step=int(done)+int(nst),restart_time_ps=float(rst),resumed=bool(int(res)),
    wall_seconds=int(wall),
    status='ok',
    files=dict(mdin='seg_%s.mdin'%s3,plumed='seg_%s.plumed.dat'%s3,
               out='seg_%s.out'%s3,rst='seg_%s.rst'%s3,nc='seg_%s.nc'%s3)))
d['segments'].sort(key=lambda x:x['segment'])
d['steps_done']=max(x['end_step'] for x in d['segments'])
d['ns_done']=round(d['steps_done']*0.002/1000,4)
json.dump(d,open(p,'w'),indent=2)
print('   PROGRESS.json: %d segment(s), %.3f ns done'%(len(d['segments']),d['ns_done']))
PY
STITCH=$SELF/h3e_stitch_colvar.py
[ -s "$STITCH" ] || STITCH=$R/scripts/h3e_stitch_colvar.py
[ -s "$STITCH" ] || { echo "FAIL: h3e_stitch_colvar.py not found next to this script"
                      echo "      (send_inputs.sh must copy it; no silent fallback)"; exit 1; }
python "$STITCH" --dir . --dt $DT
NS=$(python -c "print('%.3f'%($TOTAL*$DT/1000))")
echo "   seg_$S3 wall=$((T1-T0))s  ns/day=$(python -c "print('%.1f'%(($NSTLIM*$DT/1000)/max($((T1-T0)),1)*86400))")"
if [ "$TOTAL" -ge "$((76000000))" ] || [ "$NSTLIM" -eq 0 ]; then
  { echo "state=$STATE"; echo "direction=$DIR"; echo "steps_total=$TOTAL"; echo "ns_total=$NS"
    echo "segments=$(ls seg_*.rst | wc -l)"; echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > H3_DONE
  echo "H3 DONE  $STATE/$DIR${TAG:+ [$TAG]}  total $NS ns -> $OUT"
else
  echo "H3 LEG DONE  $STATE/$DIR${TAG:+ [$TAG]}  $NS ns of $(python -c "print('%.0f'%(76000000*$DT/1000))") ns"
  echo "   continue with the same command plus --resume"
fi
