#!/usr/bin/env bash
set -euo pipefail
md_root=${SECONDSITE_MD_ROOT:-$(cd "$(dirname "$0")" && pwd)}
md_python=${MD_PYTHON:-python3}
reference_dir=${AMMO_REFERENCE_DIR:-$md_root/refs}
reference_pdb=$reference_dir/ammo_examples__example_ptp1b__inputs__closed.pdb
mkdir -p "$md_root/audit"
legs=()
for state in open closed; do
  production=$md_root/$state/production
  snapshots=$production/snapshots
  prepared=$md_root/prepared_gmx/2e__$state
  "$md_python" "$md_root/scripts/h4c_seeds_author.py" --run "$production" --out "$snapshots" --ref "$reference_pdb"
  "$md_python" "$md_root/ops/h3x_ligand_escape.py" --dir "$production" --ref "$reference_pdb" --steps-json "$snapshots/SEEDS.json" --scratch "$md_root/contact_$state" --json "$md_root/audit/contact_$state.json"
  "$md_python" "$md_root/ops/h4p_pose_report.py" --dir "$production" --seeds-json "$snapshots/SEEDS.json" --ref-docked-parm "$prepared/dry.prmtop" --ref-docked-crd "$prepared/dry.inpcrd" --ref-equil-parm "$prepared/system.prmtop" --ref-equil-crd "$prepared/equil.rst7" --scratch "$md_root/pose_$state" --json "$md_root/audit/pose_$state.json"
  "$md_python" "$md_root/ops/h4x_image_export.py" --dir "$production" --seeds-json "$snapshots/SEEDS.json" --out "$md_root/h5_coords_$state" --scratch "$md_root/image_$state" --json "$md_root/audit/image_$state.json"
  legs+=(--leg "$state:$snapshots/SEEDS.json:$md_root/audit/contact_$state.json:$md_root/audit/pose_$state.json")
done
"$md_python" "$md_root/ops/h4s_seed_strata.py" "${legs[@]}" --json "$md_root/audit/seed_strata.json"
"$md_python" "$md_root/select_contact_seeds.py" --strata "$md_root/audit/seed_strata.json" --out "$md_root/audit/pocket.seeds"
