"""Classify docking poses by site contacts and compare matched high/low rankings."""

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def coords(value, label):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Missing or nonfinite geometry: " + label)
    return arr


def build_reference(raw, boxes):
    if raw.get("coordinates_frame") != "same reviewed docking assembly":
        raise ValueError(
            "Reference geometry is not explicitly in the reviewed receptor frame"
        )
    rxyz = coords(raw["receptor_atom_xyz"], "receptor")
    txyz = coords(raw["target_atom_xyz"], "target")
    rkeys = np.asarray(raw["receptor_atom_residue_keys"], dtype=str)
    tkeys = np.asarray(raw["target_atom_residue_keys"], dtype=str)
    if len(rkeys) != len(rxyz) or len(tkeys) != len(txyz):
        raise ValueError("Geometry residue-key count mismatch")
    primary = coords(raw["primary_xyz"], "primary reference")
    sites, coverage = {}, []
    for s in raw.get("allosteric_sites", []):
        sid = str(s["site_id"])
        if sid in sites:
            raise ValueError("Duplicate allosteric site id: " + sid)
        xyz = coords(s["xyz"], "known site " + sid)
        keys = set(s["contact_keys"])
        if len(keys) < 3:
            raise ValueError("Known site has fewer than 3 contact residues: " + sid)
        allowed = set()
        for b in boxes:
            center, half = np.asarray(b["center"]), np.asarray(b["size"]) / 2
            fraction = float(np.all(np.abs(xyz - center) <= half, axis=1).mean())
            centroid_inside = bool(np.all(np.abs(xyz.mean(0) - center) <= half))
            covered = centroid_inside and fraction >= 0.8
            if covered:
                allowed.add(b["pocket_id"])
            coverage.append(
                dict(
                    site_id=sid,
                    pocket_id=b["pocket_id"],
                    native_atom_fraction_inside=fraction,
                    native_centroid_inside=centroid_inside,
                    covered=covered,
                )
            )
        sites[sid] = dict(keys=keys, centroid=xyz.mean(0), allowed_boxes=allowed)
    strict = raw.get(
        "strict_distinct_evaluable",
        raw["reference_kind"] != "limited_catalytic_residue_reference",
    )
    if not isinstance(strict, bool):
        raise ValueError("strict_distinct_evaluable must be a JSON boolean")
    if raw["reference_kind"] == "limited_catalytic_residue_reference" and strict:
        raise ValueError(
            "Limited catalytic residues cannot define a full orthosteric separation reference"
        )
    reason = (
        raw.get(
            "strict_distinct_exclusion_reason",
            "Limited catalytic-residue reference does not define the complete orthosteric site",
        )
        if not strict
        else ""
    )
    known = raw.get("known_site_evaluable", True)
    if not isinstance(known, bool):
        raise ValueError("known_site_evaluable must be a JSON boolean")
    known_reason = (
        raw.get(
            "known_site_exclusion_reason",
            "Known-site reference is incompletely transferred or otherwise not evaluable",
        )
        if not known
        else ""
    )
    return dict(
        receptor=cKDTree(rxyz),
        receptor_keys=rkeys,
        target=cKDTree(txyz),
        target_keys=tkeys,
        primary=cKDTree(primary),
        primary_keys=set(raw["primary_contact_keys"]),
        sites=sites,
        coverage=coverage,
        reference_kind=raw["reference_kind"],
        strict_distinct_evaluable=strict,
        strict_distinct_exclusion_reason=reason,
        known_site_evaluable=known,
        known_site_exclusion_reason=known_reason,
    )


def endpoint_proteins(refs):
    strict = {uid for uid, r in refs.items() if r["strict_distinct_evaluable"]}
    known = {
        uid
        for uid, r in refs.items()
        if r["known_site_evaluable"]
        and any(s["allowed_boxes"] for s in r["sites"].values())
    }
    return strict, known


def contacted_keys(tree, keys, xyz):
    indices = {i for atom in tree.query_ball_point(xyz, 4.0) for i in atom}
    return {str(keys[i]) for i in indices}


def actual_receptor_tree(path):
    xyz = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atomtype = line.split()[-1]
            if atomtype not in ("H", "HD", "HS") and not re.fullmatch(
                r"G\d+", atomtype
            ):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return cKDTree(coords(xyz, "actual prepared receptor PDBQT"))


def classify(xyz, expected_heavy_atoms, ref, pocket_id):
    xyz = coords(xyz, "pose")
    contacts = contacted_keys(ref["receptor"], ref["receptor_keys"], xyz)
    target = contacted_keys(ref["target"], ref["target_keys"], xyz)
    atom_ok = len(xyz) == expected_heavy_atoms
    receptor_distance = float(
        ref.get("clash_receptor", ref["receptor"]).query(xyz)[0].min()
    )
    primary_distance = float(ref["primary"].query(xyz)[0].min())
    shared = contacts & ref["primary_keys"]
    valid = bool(atom_ok and receptor_distance >= 1.2 and len(target) >= 3)
    centroid = xyz.mean(0)
    known, partial, details = [], [], []
    for sid, s in ref["sites"].items():
        intersection = len(contacts & s["keys"])
        recall = intersection / len(s["keys"])
        precision = intersection / len(contacts) if contacts else 0.0
        distance = float(np.linalg.norm(centroid - s["centroid"]))
        hit = valid and intersection >= 3 and recall >= 0.25 and distance <= 5.0
        covered = pocket_id in s["allowed_boxes"]
        if hit:
            (known if covered else partial).append(sid)
        details.append(
            dict(
                site_id=sid,
                contact_recall=recall,
                contact_precision=precision,
                centroid_distance_A=distance,
                covered_box=covered,
                pose_match=bool(hit),
            )
        )
    primary_known = known if ref["known_site_evaluable"] else []
    descriptive = known + partial if not ref["known_site_evaluable"] else []
    return dict(
        valid_basic_geometry=valid,
        heavy_atom_count_matches=atom_ok,
        n_target_contact_residues=len(target),
        n_orthosteric_contact_residues=len(shared),
        min_reference_distance_A=primary_distance,
        min_receptor_distance_A=receptor_distance,
        contact_residues=";".join(sorted(contacts)),
        orthosteric_overlap_residues=";".join(sorted(shared)),
        distinct_6A=bool(valid and not shared and primary_distance >= 6.0),
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        centroid_z=float(centroid[2]),
        known_site_matches=json.dumps(primary_known),
        partial_known_site_matches=json.dumps(partial),
        non_evaluable_hypothesis_site_matches=json.dumps(descriptive),
        known_site_details=json.dumps(details),
    )


def check_seed_grid(poses):
    group = ["system_id", "pocket_id", "scoring"]
    if poses.duplicated(group + ["seed"]).any():
        raise ValueError("Repeated seed within one system/box/scoring")
    if not (poses.groupby(group).seed.nunique() == 5).all():
        raise ValueError("Expected exactly 5 seeds per system/box/scoring")
    for _, g in poses.groupby(["system_id", "pocket_id"]):
        if set(g.scoring) != {"vina", "vinardo"}:
            raise ValueError("Missing scoring function")


def medoid_cluster(g):
    xyz = g[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    if not len(xyz):
        return 0, None
    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    count = (distance <= 4.0).sum(axis=1)
    chosen = int(np.argmax(count))
    return int(count[chosen]), xyz[chosen]


def repeated_success(eligible, known=False):
    """Return supported systems; known-site seeds must agree on the SAME site."""
    if eligible.empty:
        return {
            k: set() for k in ["vina", "vinardo", "both_same_box", "spatially_stable"]
        }
    if known:
        work = eligible.copy()
        work["site_id"] = work.known_site_matches.map(json.loads)
        work = work.explode("site_id").dropna(subset=["site_id"])
    else:
        work = eligible.assign(site_id="strict_distinct6A")
    success = {
        k: set() for k in ["vina", "vinardo", "both_same_box", "spatially_stable"]
    }
    for (system, box, sid), g in work.groupby(["system_id", "pocket_id", "site_id"]):
        by = {sc: h.sort_values("seed") for sc, h in g.groupby("scoring")}
        for sc in ["vina", "vinardo"]:
            if sc in by and by[sc].seed.nunique() >= 3:
                success[sc].add(system)
        if not all(
            sc in by and by[sc].seed.nunique() >= 3 for sc in ["vina", "vinardo"]
        ):
            continue
        success["both_same_box"].add(system)
        a, ac = medoid_cluster(by["vina"])
        b, bc = medoid_cluster(by["vinardo"])
        if a >= 3 and b >= 3 and np.linalg.norm(ac - bc) <= 5:
            success["spatially_stable"].add(system)
    return success


def parse_pose(p):
    xyz = []
    score = None
    dummy = 0
    for line in Path(p).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("REMARK VINA RESULT:"):
            score = float(line.split()[3])
        if line.startswith(("ATOM  ", "HETATM")):
            typ = line.split()[-1]
            if re.fullmatch(r"G\d+", typ):
                dummy += 1
            elif typ not in ("H", "HD", "HS"):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if score is None or not xyz:
        raise ValueError("Invalid first pose: " + str(p))
    return score, np.array(xyz), dummy


def paired_stats(df, value="success", reps=10000):
    if reps < 1 or df.empty or not set(df.role) == {"candidate", "comparison"}:
        raise ValueError(
            "Paired statistics require both roles and positive resample count"
        )
    p = df.pivot(index=["uniprot", "match_id"], columns="role", values=value).astype(
        float
    )
    assert p.notna().all().all()
    delta = (p.candidate - p.comparison).values
    uids = p.index.get_level_values("uniprot").to_numpy()
    clusters = sorted(set(uids))
    sums = np.array([delta[uids == u].sum() for u in clusters])
    ns = np.array([(uids == u).sum() for u in clusters])
    rng = np.random.default_rng(20261012)
    draws = rng.integers(0, len(clusters), (reps, len(clusters)))
    boot = sums[draws].sum(axis=1) / ns[draws].sum(axis=1)
    if len(clusters) <= 12:
        null = np.array(
            [
                np.dot(s, sums) / len(delta)
                for s in itertools.product([-1, 1], repeat=len(clusters))
            ]
        )
        pvalue = float(np.mean(np.abs(null) >= abs(delta.mean()) - 1e-12))
        pmethod = "exact protein-block sign flip"
    else:
        null = rng.choice([-1, 1], size=(10000, len(clusters))) @ sums / len(delta)
        pvalue = float(
            (1 + np.sum(np.abs(null) >= abs(delta.mean()) - 1e-12)) / (len(null) + 1)
        )
        pmethod = "Monte Carlo 10000 protein-block sign flips, plus-one p"
    return dict(
        n_pairs=len(p),
        n_proteins=len(clusters),
        candidate_mean=float(p.candidate.mean()),
        comparison_mean=float(p.comparison.mean()),
        delta=float(delta.mean()),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        protein_block_signflip_two_sided_p=pvalue,
        signflip_method=pmethod,
        candidate_only=(
            int(((p.candidate == 1) & (p.comparison == 0)).sum())
            if value == "success"
            else None
        ),
        comparison_only=(
            int(((p.candidate == 0) & (p.comparison == 1)).sum())
            if value == "success"
            else None
        ),
        both=(
            int(((p.candidate == 1) & (p.comparison == 1)).sum())
            if value == "success"
            else None
        ),
        neither=(
            int(((p.candidate == 0) & (p.comparison == 0)).sum())
            if value == "success"
            else None
        ),
    )


def validate_units(units):
    if units.system_id.duplicated().any():
        raise ValueError("Repeated compound system identifier")
    for _, group in units.groupby("match_id"):
        if (
            len(group) != 2
            or set(group.role) != {"candidate", "comparison"}
            or group.uniprot.nunique() != 1
        ):
            raise ValueError(
                "Each match requires one high- and one low-ranked compound on the same protein"
            )


def endpoint_table(poses, units, refs, cutoffs=(None, -5.0)):
    validate_units(units)
    check_seed_grid(poses)
    if set(poses.system_id) != set(units.system_id):
        raise ValueError("Pose and compound populations differ")
    strict, known = endpoint_proteins(refs)
    records = []
    for endpoint, proteins in [
        ("strict_distinct6A", strict),
        ("known_allosteric_site", known),
    ]:
        population = units[units.uniprot.isin(proteins)]
        mask = (
            poses.distinct_6A
            if endpoint == "strict_distinct6A"
            else poses.known_site_matches.ne("[]")
        )
        mask = mask & poses.uniprot.isin(proteins)
        for cutoff in cutoffs:
            eligible = poses[
                mask & (True if cutoff is None else poses.score.le(cutoff))
            ]
            supported = repeated_success(
                eligible, known=endpoint == "known_allosteric_site"
            )
            for scoring, systems in supported.items():
                records.extend(
                    population.assign(
                        success=population.system_id.isin(systems),
                        endpoint=endpoint,
                        energy_cutoff="none" if cutoff is None else "%g" % cutoff,
                        scoring=scoring,
                    ).to_dict("records")
                )
    return pd.DataFrame(
        records,
        columns=list(units.columns)
        + ["success", "endpoint", "energy_cutoff", "scoring"],
    )


def strict_bool(series):
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Expected explicit boolean values")
    return values.isin(["true", "1"])


def select_compounds(frame, chemistry, proteins, count, per_protein):
    """Match top- and bottom-decile compounds before examining docking outcomes."""
    if count < 1 or per_protein < 1 or frame.OOD_Row_ID.duplicated().any():
        raise ValueError("Invalid selection limits or duplicate ranking records")
    if chemistry.ligand_chembl_id.duplicated().any():
        raise ValueError("Duplicate chemical-preparation records")
    df = frame[
        ~strict_bool(frame.has_allosteric_text)
        & ~strict_bool(frame.has_orthosteric_text)
    ].copy()
    df = df.dropna(
        subset=["uniprot", "connectivity_key", "model_score", "MW", "LogP", "TPSA"]
    )
    df = df[
        np.isfinite(df.model_score)
        & df.MW.between(150, 750)
        & df.uniprot.str.match(r"^[A-Z0-9]+$")
    ]
    df = df.sort_values(
        ["model_score", "OOD_Row_ID"], ascending=[False, True]
    ).drop_duplicates(["uniprot", "connectivity_key"])
    df["within_protein_quantile"] = df.groupby("uniprot").model_score.rank(
        pct=True, method="average"
    )
    df = df[df.uniprot.isin(proteins)]
    controls = {
        k: g for k, g in df[df.within_protein_quantile <= 0.1].groupby("uniprot")
    }
    candidates = (
        df[df.within_protein_quantile >= 0.9]
        .sort_values(
            ["uniprot", "within_protein_quantile", "model_score", "OOD_Row_ID"],
            ascending=[True, False, False, True],
        )
        .copy()
    )
    candidates["target_round"] = candidates.groupby("uniprot").cumcount()
    candidates = candidates.sort_values(["target_round", "uniprot"])
    prepared = chemistry.set_index("ligand_chembl_id").to_dict("index")
    rows, rejections, used, counts = [], [], set(), {}

    def chemical(row):
        item = prepared.get(row["ligand_chembl_id"])
        if item is None or item.get("status") != "complete":
            raise ValueError("Chemical preparation missing or unsuccessful")
        if item["computed_connectivity"] != row["connectivity_key"]:
            raise ValueError("Chemical connectivity mismatch")
        if any(
            pd.isna(item[k]) for k in ["source_formal_charge", "ph74_formal_charge"]
        ):
            raise ValueError("Missing formal charge")
        return item

    for row in candidates.to_dict("records"):
        uid = row["uniprot"]
        if counts.get(uid, 0) >= per_protein:
            continue
        try:
            c = chemical(row)
            if uid not in controls:
                raise ValueError("No bottom-decile compounds")
            pool = controls[uid]
            pool = pool[
                (abs(pool.MW - row["MW"]) <= 0.25 * row["MW"])
                & (abs(pool.LogP - row["LogP"]) <= 1.5)
                & (abs(pool.TPSA - row["TPSA"]) <= 40)
            ].copy()
            pool["match_distance"] = (
                ((pool.MW - row["MW"]) / (0.25 * row["MW"])) ** 2
                + ((pool.LogP - row["LogP"]) / 1.5) ** 2
                + ((pool.TPSA - row["TPSA"]) / 40) ** 2
            )
            chosen = None
            for other in pool.sort_values(["match_distance", "OOD_Row_ID"]).to_dict(
                "records"
            ):
                if (uid, other["connectivity_key"]) in used or other[
                    "model_score"
                ] >= row["model_score"]:
                    continue
                try:
                    cc = chemical(other)
                except ValueError:
                    continue
                if any(
                    c[k] != cc[k]
                    for k in ["source_formal_charge", "ph74_formal_charge"]
                ):
                    continue
                chosen = dict(other, **cc)
                break
            if chosen is None:
                raise ValueError(
                    "No unused property- and charge-matched bottom-decile compound"
                )
            mid = "pair_%03d" % (len(rows) // 2 + 1)
            for role, item in [("candidate", dict(row, **c)), ("comparison", chosen)]:
                rows.append(
                    dict(item, match_id=mid, role=role, system_id=mid + "_" + role)
                )
                used.add((uid, item["connectivity_key"]))
            counts[uid] = counts.get(uid, 0) + 1
            if len(rows) // 2 >= count:
                break
        except ValueError as exc:
            rejections.append(
                dict(OOD_Row_ID=row["OOD_Row_ID"], uniprot=uid, reason=str(exc))
            )
    if not rows:
        raise ValueError("No matched top/bottom compounds")
    return pd.DataFrame(rows), pd.DataFrame(
        rejections, columns=["OOD_Row_ID", "uniprot", "reason"]
    )


def summarize_endpoints(frame, reps=10000, family_size=16):
    """Protein-cluster uncertainty; Holm adjustment across cohort/model conditions."""
    frame = frame.copy()
    frame["success"] = strict_bool(frame.success)
    grouping = ["cohort", "model", "endpoint", "energy_cutoff", "scoring"]
    if "subset" in frame:
        grouping.append("subset")
    rows = []
    for key, group in frame.groupby(grouping, dropna=False):
        validate_units(group[["system_id", "match_id", "uniprot", "role"]])
        rows.append(dict(zip(grouping, key), **paired_stats(group, reps=reps)))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["holm_p"] = np.nan
    for _, group in result.groupby(
        [c for c in grouping if c not in ("cohort", "model")], dropna=False
    ):
        if len(group) > family_size:
            raise ValueError("More conditions than the specified testing family")
        ordered = group.sort_values("protein_block_signflip_two_sided_p")
        adjusted = np.maximum.accumulate(
            ordered.protein_block_signflip_two_sided_p.to_numpy()
            * (family_size - np.arange(len(ordered)))
        )
        result.loc[ordered.index, "holm_p"] = np.minimum(1, adjusted)
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_jobs(args):
    jobs = pd.read_csv(args.jobs, sep="\t")
    units = pd.read_csv(args.systems, sep="\t")
    validate_units(units)
    if jobs.job_id.duplicated().any() or set(jobs.system_id) != set(units.system_id):
        raise ValueError("The complete job plan and compound table must agree")
    raw = json.loads(args.references.read_text())
    refs = {}
    for uid in units.uniprot.unique():
        item = raw[uid]
        refs[uid] = build_reference(item, item["boxes"])
        refs[uid]["clash_receptor"] = actual_receptor_tree(
            args.root / item["prepared_receptor"]
        )
    identity = units.set_index("system_id").to_dict("index")
    for mid, group in units.groupby("match_id"):
        uid = group.iloc[0].uniprot
        expected = {
            (s, b["pocket_id"], scoring, int(seed))
            for s in group.system_id
            for b in raw[uid]["boxes"]
            for scoring in ["vina", "vinardo"]
            for seed in args.seeds
        }
        planned = jobs[jobs.system_id.isin(group.system_id)]
        actual = list(
            zip(
                planned.system_id,
                planned.pocket_id,
                planned.scoring,
                planned.seed.astype(int),
            )
        )
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Incomplete planned box/scoring/seed grid: " + str(mid))
    rows = []
    for row in jobs.to_dict("records"):
        unit = identity[row["system_id"]]
        path = args.root / row["pose_path"]
        if sha256(path) != row["pose_sha256"]:
            raise ValueError("Pose checksum mismatch: " + str(row["job_id"]))
        score, xyz, dummy = parse_pose(path)
        if not np.isfinite(score):
            raise ValueError("Nonfinite docking score")
        fields = {
            k: row[k] for k in ["job_id", "system_id", "pocket_id", "scoring", "seed"]
        }
        fields.update({k: unit[k] for k in ["match_id", "uniprot", "role"]})
        fields.update(score=score, ignored_dummy_atoms=dummy)
        fields.update(
            classify(
                xyz, int(unit["heavy_atoms"]), refs[unit["uniprot"]], row["pocket_id"]
            )
        )
        rows.append(fields)
    poses = pd.DataFrame(rows)
    endpoints = endpoint_table(poses, units, refs, [None] + args.energy_cutoffs)
    args.output.mkdir(parents=True, exist_ok=False)
    poses.to_csv(args.output / "pose_sites.tsv.gz", sep="\t", index=False)
    endpoints.to_csv(args.output / "endpoints.tsv", sep="\t", index=False)
    coverage = [dict(uniprot=u, **c) for u, r in refs.items() for c in r["coverage"]]
    pd.DataFrame(coverage).to_csv(
        args.output / "site_search_coverage.tsv", sep="\t", index=False
    )
    summarize_endpoints(endpoints, args.replicates).to_csv(
        args.output / "statistics.tsv", sep="\t", index=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection_parser = sub.add_parser(
        "select", help="Select property-matched top/bottom-decile compounds"
    )
    for name in ["rankings", "chemistry", "proteins", "output"]:
        selection_parser.add_argument("--" + name, type=Path, required=True)
    selection_parser.add_argument("--count", type=int, default=1000)
    selection_parser.add_argument("--per-protein", type=int, required=True)
    classify_parser = sub.add_parser(
        "classify", help="Classify first poses using reviewed reference geometry"
    )
    for name in ["jobs", "systems", "references", "root", "output"]:
        classify_parser.add_argument("--" + name, type=Path, required=True)
    classify_parser.add_argument(
        "--seeds", type=int, nargs=5, default=list(range(20261012, 20261017))
    )
    classify_parser.add_argument(
        "--energy-cutoffs", type=float, nargs="*", default=[-5.0]
    )
    classify_parser.add_argument("--replicates", type=int, default=10000)
    summary_parser = sub.add_parser(
        "summarize",
        help="Compare archived endpoints across training datasets and models",
    )
    summary_parser.add_argument("--endpoints", type=Path, nargs="+", required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.add_argument("--replicates", type=int, default=10000)
    summary_parser.add_argument("--family-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "select":
        selected, rejected = select_compounds(
            pd.read_csv(args.rankings, sep="\t"),
            pd.read_csv(args.chemistry, sep="\t"),
            set(args.proteins.read_text().split()),
            args.count,
            args.per_protein,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        selected.to_csv(args.output / "selected_pairs.tsv", sep="\t", index=False)
        rejected.to_csv(args.output / "unmatched_candidates.tsv", sep="\t", index=False)
    elif args.command == "classify":
        if len(set(args.seeds)) != 5:
            raise ValueError("Five distinct seeds are required")
        classify_jobs(args)
    else:
        frames = [
            pd.read_csv(p, sep="\t", dtype={"energy_cutoff": str})
            for p in args.endpoints
        ]
        result = summarize_endpoints(
            pd.concat(frames, ignore_index=True), args.replicates, args.family_size
        )
        args.output.mkdir(parents=True, exist_ok=False)
        result.to_csv(args.output / "statistics.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
