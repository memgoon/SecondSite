#!/bin/bash
set -eo pipefail
SEED=""; NS=50; FORCE=0; ACCEPT_UNVERIFIED=0; CHECK_ONLY=0
while [ $# -gt 0 ]; do case "$1" in
  --seed)  SEED=$2; shift 2;;
  --ns)    NS=$2;   shift 2;;
  --force) FORCE=1; shift;;
  --accept-unverified-dry) ACCEPT_UNVERIFIED=1; shift;;
  --check-only) CHECK_ONLY=1; shift;;
  *) echo "unknown argument: $1"; exit 2;;
esac; done
[ -n "$SEED" ] || { echo "usage: h5b_seeded_md.sh --seed <dir> [--ns 50] [--force]"; exit 2; }
SEED=$(cd "$SEED" && pwd)
SELF=$(cd "$(dirname "$0")" && pwd)
R=${SECONDSITE_MD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
VERIFY=$SELF/h5v_verify.py; [ -s "$VERIFY" ] || VERIFY=$R/scripts/h5v_verify.py
[ -s "$VERIFY" ] || { echo "h5b: h5v_verify.py not found next to this script"; exit 1; }

DT=0.002; REPORT=5000
cd "$SEED"
[ -r "${GMXRC:-/opt/gromacs-2020.2/bin/GMXRC}" ] && source "${GMXRC:-/opt/gromacs-2020.2/bin/GMXRC}" || true
export PATH=/opt/pmemd26/bin:/opt/conda/bin:$PATH
NSTLIM=$(python3 -c "print(int(round($NS*1000/$DT)))")
EXP_FRAMES=$((NSTLIM/REPORT))

[ -f PREP_DONE ] || { echo "h5b: $SEED has no PREP_DONE - run h5_seed_prep.sh first"; exit 1; }
for f in system.prmtop system_dry.prmtop seed_equil.rst7; do
  [ -s "$f" ] || { echo "h5b: missing $f"; exit 1; }
done

build_mdin(){   # $1 = destination
cat > "$1" <<EOF
Production.
 &cntrl
  ig=-1,
  ntx=1,
  ntxo=1,
  ntpr=$REPORT,
  ntwr=$REPORT,
  ntwx=$REPORT,
  irest=0,
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
 /
EOF
}
MDTMP=$(mktemp "${TMPDIR:-/tmp}/h5b_mdin.XXXXXX")
trap 'rm -f "$MDTMP"' EXIT
build_mdin "$MDTMP"
write_mdin(){ cp -f "$MDTMP" production.mdin; }
SIG="ns=$NS nstlim=$NSTLIM ntwx=$REPORT prmtop=$(md5sum system.prmtop | cut -d' ' -f1) rst=$(md5sum seed_equil.rst7 | cut -d' ' -f1) mdin=$(md5sum "$MDTMP" | cut -d' ' -f1)"

sig_of(){ grep -m1 '^signature=' "$1" 2>/dev/null | sed 's/^signature=//'; }
explain_mismatch(){
  local was=$1
  echo "  recorded: $was"
  echo "  requested: $SIG"
  for k in ns nstlim ntwx prmtop rst mdin; do
    local a b
    a=$(echo "$was" | tr ' ' '\n' | grep "^$k=" || true)
    b=$(echo "$SIG" | tr ' ' '\n' | grep "^$k=" || true)
    [ "$a" != "$b" ] && echo "    differs: $k  $a -> $b"
  done
}
stash(){
  local n=1
  while [ -e "attempt_$(printf %02d $n)" ]; do n=$((n+1)); done
  local A="attempt_$(printf %02d $n)"; mkdir -p "$A"
  local moved=0
  for f in production.nc production_dry.nc production.rst production.out \
           production.info production.mdin MD_DONE SEED_DONE \
           h5v_md.log h5v_dry.log h5v_dry.json h5v_cpptraj.log h5v_dry.in; do
    [ -e "$f" ] && { mv "$f" "$A/"; moved=$((moved+1)); }
  done
  echo "  moved $moved leftover file(s) to $A/ (nothing deleted)"
  write_mdin
}

dry_md5(){ md5sum production_dry.nc 2>/dev/null | cut -d' ' -f1; }
field(){ grep -m1 "^$2=" "$1" 2>/dev/null | sed "s/^$2=//"; }

dry_artefact_ok(){
  local rec=$1 want_md5 want_bytes coords
  want_md5=$(field "$rec" dry_md5); want_bytes=$(field "$rec" dry_bytes)
  coords=$(field "$rec" wet_dry_coords_verified)
  [ -s production_dry.nc ] || { echo "   production_dry.nc is missing"; return 1; }
  [ -n "$want_bytes" ] || { echo "   the record carries no dry_bytes, so it cannot be trusted"; return 1; }
  [ -n "$want_md5" ]   || { echo "   the record carries no dry_md5, so it cannot be trusted"; return 1; }
  [ "$coords" = yes ] || {
    echo "   the record does not say the wet->dry coordinate comparison passed"
    echo "   (wet_dry_coords_verified=${coords:-absent})"; return 1; }
  if [ "$(stat -c%s production_dry.nc)" != "$want_bytes" ]; then
    echo "   production_dry.nc is $(stat -c%s production_dry.nc) bytes, recorded $want_bytes"; return 1; fi
  if [ "$(dry_md5)" != "$want_md5" ]; then
    echo "   production_dry.nc md5 $(dry_md5) does not match the recorded $want_md5"; return 1; fi
  return 0
}
stash_one(){   # preserve a rejected artefact instead of overwriting it
  local f=$1 n=1
  while [ -e "rejected_$(printf %02d $n)_$f" ]; do n=$((n+1)); done
  mv "$f" "rejected_$(printf %02d $n)_$f"
  echo "   kept the rejected $f as rejected_$(printf %02d $n)_$f"
}
make_dry(){
  cat > h5v_dry.in <<EOF
parm system.prmtop
trajin production.nc
strip :WAT,:SOL,:HOH,:NA,:CL,:Na+,:Cl-,:CLA,:POT,:SOD
trajout production_dry.nc netcdf
go
quit
EOF
  cpptraj -i h5v_dry.in > h5v_cpptraj.log 2>&1 || return 1
  [ -s production_dry.nc ] || return 1
  return 0
}
coords_evidence(){   # from h5v_dry.json, written by the verifier itself
  python3 - <<'EOF'
import json,os
try:
    e=json.load(open('h5v_dry.json'))
except Exception:
    print('no'); print(''); raise SystemExit(0)
print('yes' if e.get('wet_dry_coords_verified') else 'no')
d=e.get('wet_dry_max_dev_A')
print('' if d is None else '%.3e'%d)
EOF
}
write_seed_done(){   # atomically, BEFORE anything is deleted
  local wet_state=$1 cv dev
  cv=$(coords_evidence | sed -n 1p); dev=$(coords_evidence | sed -n 2p)
  [ "$ACCEPT_UNVERIFIED" = 1 ] && [ "$cv" != yes ] && cv=accepted_unverified
  { echo "signature=$SIG"; echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "ns=$NS"; echo "nstlim=$NSTLIM"; echo "ntwx=$REPORT"; echo "frames=$EXP_FRAMES"
    echo "wall_seconds=${WALL:-unknown}"
    echo "dry_bytes=$(stat -c%s production_dry.nc)"; echo "dry_md5=$(dry_md5)"
    echo "wet_dry_coords_verified=$cv"; echo "wet_dry_max_dev_A=${dev:-none}"
    echo "wet_present=$wet_state"
    echo "velocities=regenerated (ntx=1/irest=0, BSS Production restart=False)"
    echo "verified_by=h5v_verify.py --mode md and --mode dry; logs h5v_md.log h5v_dry.log"
  } > SEED_DONE.tmp
  mv -f SEED_DONE.tmp SEED_DONE
}

if [ "$CHECK_ONLY" = 1 ]; then
  if [ -f SEED_DONE ] && [ "$(sig_of SEED_DONE)" = "$SIG" ] && dry_artefact_ok SEED_DONE >/dev/null 2>&1; then
    echo "complete"; exit 0
  fi
  echo "incomplete"; exit 3
fi

SKIP_MD=0
if [ -f SEED_DONE ] && [ "$FORCE" = 0 ]; then
  if [ "$(sig_of SEED_DONE)" = "$SIG" ]; then
    if dry_artefact_ok SEED_DONE; then
      echo "h5b: $SEED is already complete for this request and production_dry.nc verifies"
      sed 's/^/   /' SEED_DONE; exit 0
    fi
    echo "h5b: SEED_DONE matches this request but its dry artefact does not:"
    if [ -s production.nc ]; then
      echo "   the wet trajectory is still here, so the drying step will be redone"
      [ -e production_dry.nc ] && stash_one production_dry.nc
      mv -f SEED_DONE "SEED_DONE.superseded.$(date -u +%Y%m%dT%H%M%SZ)"
      SKIP_MD=1
    else
      echo "REFUSING: neither a valid dry trajectory nor the wet one survives, so this"
      echo "  seed cannot be completed without re-running its 50 ns. Use --force to redo."
      exit 3
    fi
  else
    echo "REFUSING: SEED_DONE exists but was produced by a DIFFERENT request."
    explain_mismatch "$(sig_of SEED_DONE)"
    echo "  This is how a short smoke test would otherwise be mistaken for a 50 ns run."
    echo "  Use --force to set the old attempt aside and redo, or run the test in a copy."
    exit 3
  fi
fi

if [ "$FORCE" = 1 ]; then
  stash
elif [ "$SKIP_MD" = 0 ] && [ -f MD_DONE ]; then
  if [ "$(sig_of MD_DONE)" != "$SIG" ]; then
    echo "REFUSING: MD_DONE exists but does not match this request."
    explain_mismatch "$(sig_of MD_DONE)"
    echo "  Use --force to set it aside and redo."
    exit 3
  fi
  if [ -s production.nc ]; then
    echo "## 0/3 MD_DONE matches this request and production.nc is present"
    echo "       re-verifying it rather than recomputing $NS ns"
    if python3 "$VERIFY" --mode md --dir . --nstlim "$NSTLIM" --ntwx "$REPORT" 2>&1 | tee h5v_md.log | sed 's/^/   /'; then
      SKIP_MD=1
    else
      echo "   the stored MD does not verify; setting it aside and recomputing"
      stash
    fi
  elif [ -s production_dry.nc ]; then
    echo "## 0/3 MD_DONE matches and the wet file is gone, but a dry copy exists"
    if ! python3 "$VERIFY" --mode dry --dir . --nstlim "$NSTLIM" --ntwx "$REPORT" 2>&1 | tee h5v_dry.log | sed 's/^/   /'; then
      echo "REFUSING: the dry copy does not verify and the wet one is gone."
      echo "  This seed needs its 50 ns re-running: use --force."
      exit 3
    fi
    if [ "$ACCEPT_UNVERIFIED" = 1 ]; then
      write_seed_done no
      echo "   accepted WITHOUT the wet->dry coordinate comparison, at your instruction;"
      echo "   SEED_DONE records wet_dry_coords_verified=accepted_unverified"
      exit 0
    fi
    echo "REFUSING: the dry copy passes the structural checks, but its wet original is"
    echo "  gone and no record says the wet->dry coordinate comparison ever passed, so"
    echo "  it cannot be shown to be this run's trajectory. Either re-run the 50 ns"
    echo "  (--force) or accept it deliberately (--accept-unverified-dry), which is"
    echo "  recorded in SEED_DONE."
    exit 3
  else
    echo "REFUSING: MD_DONE exists but neither production.nc nor production_dry.nc does."
    echo "  Use --force to redo."
    exit 3
  fi
elif [ "$SKIP_MD" = 0 ] && { [ -e production.nc ] || [ -e production.out ]; }; then
  echo "## 0/3 leftovers from an unfinished attempt"
  stash
fi

if [ "$SKIP_MD" = 0 ]; then
  write_mdin
  echo "## 1/3 pmemd.cuda  $NS ns = $NSTLIM steps, ntwx=$REPORT -> $EXP_FRAMES frames"
  nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader | sed 's/^/   /'
  T0=$(date +%s)
  pmemd.cuda -O -i production.mdin -p system.prmtop -c seed_equil.rst7 \
    -o production.out -r production.rst -x production.nc -inf production.info
  T1=$(date +%s)
  WALL=$((T1-T0))
  echo "   wall ${WALL}s -> $(python3 -c "print('%.1f'%($NS/max($WALL,1)*86400))") ns/day"
  echo "## 2/3 verify the wet run (frame and atom counts from the file header)"
  python3 "$VERIFY" --mode md --dir . --nstlim "$NSTLIM" --ntwx "$REPORT" 2>&1 | tee h5v_md.log | sed 's/^/   /'
  { echo "signature=$SIG"; echo "md_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_seconds=$WALL"; echo "frames=$EXP_FRAMES"; } > MD_DONE
else
  WALL=$(field MD_DONE wall_seconds)
  echo "## 1-2/3 skipped: the 50 ns MD is already done and verified (wall ${WALL:-?}s)"
fi

echo "## 3/3 dry the trajectory, verify it against the wet one, then record and clean up"
TRY=0
while :; do
  TRY=$((TRY+1))
  if [ ! -s production_dry.nc ]; then
    make_dry || { echo "   cpptraj failed; the wet file is KEPT"
                  tail -20 h5v_cpptraj.log | sed 's/^/     /'; exit 1; }
  fi
  if python3 "$VERIFY" --mode dry --dir . --nstlim "$NSTLIM" --ntwx "$REPORT" 2>&1 | tee h5v_dry.log | sed 's/^/   /'; then
    break
  fi
  if [ "$TRY" -ge 2 ]; then
    echo "   dry verification failed twice; the wet production.nc is KEPT for another attempt"
    exit 1
  fi
  echo "   dry verification failed; rebuilding it from the intact wet trajectory"
  stash_one production_dry.nc
done

write_seed_done yes
WETSZ=$(stat -c%s production.nc 2>/dev/null || echo 0)
if [ "$WETSZ" != 0 ]; then
  rm -f production.nc
  write_seed_done no
  echo "   recorded, then deleted the wet $(python3 -c "print('%.2f'%($WETSZ/1e9))") GB; dry $(python3 -c "print('%.2f'%($(stat -c%s production_dry.nc)/1e9))") GB kept"
else
  write_seed_done no
  echo "   wet file was already gone; dry $(python3 -c "print('%.2f'%($(stat -c%s production_dry.nc)/1e9))") GB kept"
fi
rm -f h5v_dry.in
echo "## SEED_DONE"
