#!/usr/bin/env python
"""Shared ENM helpers for ASD/AlloBench x PLINDER analyses.

Residues are keyed by UniProt number.  Structure mapping uses SIFTS label_seq,
matching the convention in the existing metric scripts.
"""
import ast
import csv
import json
import math
import os
import re
from collections import defaultdict

import gemmi
import numpy as np
from prody import ANM, calcSqFlucts, confProDy

confProDy(verbosity="none")

STRUCT = "data/structures"
SIFTS = "analysis/sifts"


def parse_list(value):
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return ast.literal_eval(value)


def load_ground_truth(path="analysis/ground_truth_allobench.csv"):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["system_id"]] = row
    return out


def load_scale_sm_apo(path="analysis/scale_results.csv"):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("sm_apo"):
                out[row["system_id"]] = row["sm_apo"]
    return out


def sifts_segments(pdb, uniprot):
    fp = f"{SIFTS}/{pdb}.json"
    if not os.path.exists(fp):
        return []
    data = json.load(open(fp))
    return [
        s
        for s in data.get(uniprot, [])
        if s.get("label_start") is not None and s.get("label_end") is not None and s.get("unp_start") is not None
    ]


def _label_to_uniprot(chain_segments, label_seq):
    for seg in chain_segments:
        if seg["label_start"] <= label_seq <= seg["label_end"]:
            return int(label_seq - (seg["label_start"] - seg["unp_start"]))
    return None


def read_structure(pdb):
    fp = f"{STRUCT}/{pdb.lower()}.cif"
    if not os.path.exists(fp) or os.path.getsize(fp) < 500:
        return None
    try:
        st = gemmi.read_structure(fp)
        st.setup_entities()
        return st
    except Exception:
        return None


def load_protein_nodes(pdb, uniprot, want_chain=None):
    """Return dict with UniProt residue ids, CA coords, auth map and model.

    If want_chain is supplied, prefer that chain; otherwise prefer the first
    chain with a SIFTS mapping for the requested UniProt.
    """
    pdb = pdb.lower()
    st = read_structure(pdb)
    if st is None:
        return None
    model = st[0]
    by_chain = defaultdict(list)
    for seg in sifts_segments(pdb, uniprot):
        by_chain[seg["chain"]].append(seg)
    if not by_chain:
        return None

    all_auth2unp = {}
    for cname, segs in by_chain.items():
        ch = model.find_chain(cname)
        if ch is None:
            continue
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if not (info and info.is_amino_acid()) or res.label_seq is None:
                continue
            unp = _label_to_uniprot(segs, res.label_seq)
            if unp is not None:
                all_auth2unp[(cname, int(res.seqid.num))] = unp

    chain_order = []
    if want_chain:
        chain_order.append(want_chain)
    chain_order.extend([c for c in by_chain if c not in chain_order])

    for cname in chain_order:
        if cname not in by_chain:
            continue
        ch = model.find_chain(cname)
        if ch is None:
            continue
        residues = []
        coords = []
        auth2unp = {}
        respos = {}
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if not (info and info.is_amino_acid()) or res.label_seq is None:
                continue
            unp = _label_to_uniprot(by_chain[cname], res.label_seq)
            if unp is None:
                continue
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            residues.append(unp)
            coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
            auth2unp[(cname, int(res.seqid.num))] = unp
            respos[unp] = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=float)
        if residues:
            all_auth2unp.update(auth2unp)
            return {
                "pdb": pdb,
                "chain": cname,
                "structure": st,
                "model": model,
                "residues": residues,
                "coords": np.asarray(coords, dtype=float),
                "auth2unp": all_auth2unp,
                "respos": respos,
            }
    return None


def ca_rmsd(nodes_a, nodes_b):
    common = [u for u in nodes_a["residues"] if u in nodes_b["respos"]]
    if len(common) < 8:
        return math.nan
    p = np.asarray([nodes_a["respos"][u] for u in common], dtype=float)
    q = np.asarray([nodes_b["respos"][u] for u in common], dtype=float)
    p0 = p - p.mean(axis=0)
    q0 = q - q.mean(axis=0)
    h = q0.T @ p0
    try:
        u, s, vt = np.linalg.svd(h)
    except np.linalg.LinAlgError:
        return math.nan
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    q_fit = q0 @ rot
    return float(np.sqrt(np.mean(np.sum((p0 - q_fit) ** 2, axis=1))))


def choose_best_apo(work_item, holo_nodes=None, sm_apo=None):
    apos = work_item.get("apos", [])
    if sm_apo and sm_apo in apos:
        aid = sm_apo
        pdb, chain = split_apo_id(aid)
        nodes = load_protein_nodes(pdb, work_item["uniprot"], chain)
        if nodes is not None:
            return aid, nodes, math.nan

    if holo_nodes is None:
        holo_nodes = load_protein_nodes(work_item["holo_pdb"], work_item["uniprot"], work_item.get("holo_chain"))
    if holo_nodes is None:
        return None, None, math.nan
    best = (None, None, math.inf)
    for aid in apos:
        pdb, chain = split_apo_id(aid)
        nodes = load_protein_nodes(pdb, work_item["uniprot"], chain)
        if nodes is None:
            continue
        rmsd = ca_rmsd(holo_nodes, nodes)
        if rmsd == rmsd and rmsd < best[2]:
            best = (aid, nodes, rmsd)
    if best[1] is None:
        return None, None, math.nan
    return best


def split_apo_id(apo_id):
    if "_" in apo_id:
        pdb, chain = apo_id.split("_", 1)
        return pdb.lower(), chain
    return apo_id.lower(), None


def allo_uniprot_set(gt_row, auth2unp):
    allo = set()
    text = gt_row.get("allo_res", "") if gt_row else ""
    for chain, num in re.findall(r"([A-Za-z0-9]+)-[A-Z]{3}-(-?\d+)", text):
        unp = auth2unp.get((chain, int(num)))
        if unp is not None:
            allo.add(unp)
    return allo


def ligand_centroids(model, ccd, chain_hint=None, near_points=None, max_keep=2):
    """Return coarse ligand centroids for CCD, preferring matching/near chain."""
    ccd = (ccd or "").upper()
    centers = []
    for ch in model:
        for res in ch:
            if res.name.upper() != ccd:
                continue
            pts = []
            for atom in res:
                if atom.element.name != "H":
                    pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
            if not pts:
                continue
            cen = np.asarray(pts, dtype=float).mean(axis=0)
            near = math.inf
            if near_points is not None and len(near_points):
                near = float(np.min(np.linalg.norm(np.asarray(near_points) - cen, axis=1)))
            centers.append((ch.name == chain_hint, near, ch.name, cen))
    if not centers:
        return np.empty((0, 3), dtype=float)
    centers.sort(key=lambda x: (not x[0], x[1]))
    chosen = [c[-1] for c in centers[:max_keep] if c[0] or c[1] <= 12.0]
    if not chosen:
        chosen = [centers[0][-1]]
    return np.asarray(chosen, dtype=float)


def ligand_contact_uniprot_set(model, ccd, auth2unp, cutoff=8.0):
    """Map protein residues near CCD ligand heavy atoms to UniProt ids.

    This is a fallback source-site definition for negative systems that do not
    have an annotated allosteric site.  It uses author chain/number from the
    structure and the SIFTS-derived auth2unp map.
    """
    ccd = (ccd or "").upper()
    lig_pts = []
    for ch in model:
        for res in ch:
            if res.name.upper() != ccd:
                continue
            for atom in res:
                if atom.element.name != "H":
                    lig_pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not lig_pts:
        return set()
    lig_pts = np.asarray(lig_pts, dtype=float)
    out = set()
    for ch in model:
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if not (info and info.is_amino_acid()):
                continue
            unp = auth2unp.get((ch.name, int(res.seqid.num)))
            if unp is None:
                continue
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            p = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=float)
            if float(np.min(np.linalg.norm(lig_pts - p, axis=1))) <= cutoff:
                out.add(unp)
    return out


def calc_anm_msf(coords, n_modes=80, cutoff=15.0):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 5:
        return None
    anm = ANM("anm")
    try:
        anm.buildHessian(coords, cutoff=cutoff, gamma=1.0)
        max_modes = max(1, min(int(n_modes), 3 * len(coords) - 6))
        anm.calcModes(n_modes=max_modes, zeros=False, turbo=True)
        return np.asarray(calcSqFlucts(anm), dtype=float)
    except Exception:
        return None


def build_anm(coords, n_modes=80, cutoff=15.0):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 5:
        return None
    anm = ANM("prs")
    try:
        anm.buildHessian(coords, cutoff=cutoff, gamma=1.0)
        max_modes = max(1, min(int(n_modes), 3 * len(coords) - 6))
        anm.calcModes(n_modes=max_modes, zeros=False, turbo=True)
        return anm
    except Exception:
        return None


def mean_or_nan(values):
    vals = [float(v) for v in values if v == v]
    return float(np.mean(vals)) if vals else math.nan


def median_or_nan(values):
    vals = [float(v) for v in values if v == v]
    return float(np.median(vals)) if vals else math.nan


def std_or_nan(values):
    vals = [float(v) for v in values if v == v]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan


def distance_matched_sets(residues, coords, target, exclude, n_pick, n_sets=32, seed=1):
    """Sample non-site residue index sets with target-distance matching."""
    residues = list(residues)
    coords = np.asarray(coords, dtype=float)
    target = np.asarray(target, dtype=float)
    rng = np.random.default_rng(seed)
    idx_by_res = {u: i for i, u in enumerate(residues)}
    target_idx = [idx_by_res[u] for u in target if u in idx_by_res]
    pool_idx = [i for i, u in enumerate(residues) if u not in exclude]
    if not target_idx or len(pool_idx) < n_pick:
        return []
    center = coords[target_idx].mean(axis=0)
    target_d = np.linalg.norm(coords[target_idx] - center, axis=1)
    pool_d = np.linalg.norm(coords[pool_idx] - center, axis=1)
    sets = []
    for _ in range(n_sets):
        available = set(pool_idx)
        chosen = []
        for d in rng.permutation(target_d):
            if not available:
                break
            cand = np.fromiter(available, dtype=int)
            cd = np.linalg.norm(coords[cand] - center, axis=1)
            order = np.argsort(np.abs(cd - d))
            take_from = order[: min(8, len(order))]
            pick = int(cand[int(rng.choice(take_from))])
            chosen.append(pick)
            available.remove(pick)
        if len(chosen) >= max(1, min(n_pick, len(target_idx))):
            sets.append(chosen)
    return sets


def summarize_rows(rows, metric):
    lines = []
    for feature in ("Activator", "Inhibitor", "Regulator"):
        vals = [float(r[metric]) for r in rows if r.get("feature") == feature and r.get(metric) not in ("", None)]
        vals = [v for v in vals if v == v]
        prot = defaultdict(list)
        for r in rows:
            if r.get("feature") != feature or r.get(metric) in ("", None):
                continue
            try:
                v = float(r[metric])
            except Exception:
                continue
            if v == v:
                prot[r["uniprot"]].append(v)
        pmed = [float(np.median(v)) for v in prot.values() if v]
        lines.append(
            "{}: systems={} median={:.4g}; proteins={} protein_median={:.4g}".format(
                feature,
                len(vals),
                float(np.median(vals)) if vals else math.nan,
                len(pmed),
                float(np.median(pmed)) if pmed else math.nan,
            )
        )
    return lines
