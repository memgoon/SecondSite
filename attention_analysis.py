"""Residue attention enrichment and ligand-dependent site attention."""

import argparse
import json
import hashlib
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon, spearmanr

B = 10000


def fractional_top(a, mask, fraction):
    k = max(1, int(np.ceil(len(a) * fraction)))
    threshold = np.sort(a)[-k]
    above = a > threshold
    equal = a == threshold
    tp = float(mask[above].sum()) + (k - above.sum()) * float(mask[equal].mean())
    return dict(precision=tp / k, recall=tp / mask.sum(), k=k)


def metrics(a, mask):
    a = np.asarray(a, float)
    mask = np.asarray(mask, bool)
    assert (
        a.shape == mask.shape
        and np.isfinite(a).all()
        and (a >= 0).all()
        and np.isclose(a.sum(), 1, atol=1e-6)
    )
    n = len(a)
    m = int(mask.sum())
    if m == 0 or m == n:
        return dict(
            status="no_positive_residues" if m == 0 else "no_negative_residues",
            n_residues=n,
            n_site=m,
        )
    ranks = rankdata(a, method="average")
    auc = float((ranks[mask].sum() - m * (m + 1) / 2) / (m * (n - m)))
    order = np.argsort(-a, kind="stable")
    y = mask[order]
    s = a[order]
    end = np.r_[np.flatnonzero(np.diff(s)), n - 1]
    tp = np.cumsum(y)[end]
    ap = float(np.sum(np.diff(np.r_[0.0, tp / m]) * tp / (end + 1)))
    prop = float(a[mask].sum())
    rotations = np.fft.ifft(
        np.fft.fft(a) * np.conj(np.fft.fft(mask.astype(float)))
    ).real
    p = float(np.mean(rotations >= prop - 1e-10))
    result = dict(
        status="evaluated",
        n_residues=n,
        n_site=m,
        site_fraction=m / n,
        attention_proportion=prop,
        attention_enrichment=prop / (m / n),
        residue_auroc=auc,
        residue_auprc=ap,
        auprc_over_prevalence=ap / (m / n),
        circular_shift_p=p,
        normalized_entropy=float(
            -np.sum(a * np.log(np.maximum(a, 1e-300))) / np.log(n)
        ),
    )
    for f in [0.05, 0.10]:
        r = fractional_top(a, mask, f)
        tag = str(int(f * 100))
        result["top" + tag + "_precision"] = r["precision"]
        result["top" + tag + "_recall"] = r["recall"]
    return result


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def summary(x, cl, tag, unit):
    x = np.asarray(x, float)
    lx = np.log(x)
    f = pd.DataFrame(dict(v=lx, c=np.asarray(cl))).groupby("c").v.agg(["sum", "size"])
    rng = np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))
    w = rng.multinomial(len(f), np.ones(len(f)) / len(f), size=B)
    bs = (w @ f["sum"].to_numpy()) / (w @ f["size"].to_numpy())
    p = float(wilcoxon(lx).pvalue)
    fam = (
        float(wilcoxon(pd.Series(lx).groupby(np.asarray(cl)).mean()).pvalue)
        if unit == "protein" and len(f) > 5
        else np.nan
    )
    return dict(
        n=len(x),
        clusters=len(f),
        geometric_mean=float(np.exp(lx.mean())),
        ci_low=float(np.exp(np.quantile(bs, 0.025))),
        ci_high=float(np.exp(np.quantile(bs, 0.975))),
        median=float(np.median(x)),
        below_1=int((x < 1).sum()),
        wilcoxon_p_vs_1=p,
        stars=stars(p),
        cluster_level_p_vs_1=fam,
    )


def analyze(args):
    R = args.output
    R.mkdir(parents=True, exist_ok=False)
    pairs = json.loads(args.pairs.read_text())
    allo = json.loads(args.allosteric_masks.read_text())
    maps = np.load(args.maps, allow_pickle=False)
    site = {}
    src = {}
    for p in pairs:
        if (
            p["binary_label"] == 0
            and p["mask_status"]["cross_structure_consensus60"] == "available"
        ):
            site[p["pair_id"]] = np.array(
                p["masks"]["cross_structure_consensus60"], bool
            )
            src[p["pair_id"]] = "BioLiP_contact_consensus"
        elif p["binary_label"] == 1 and allo[p["pair_id"]]["status"] == "available":
            site[p["pair_id"]] = np.array(allo[p["pair_id"]]["mask"], bool)
            src[p["pair_id"]] = "AlloBench_allosteric_site"
    rows = []
    for m in ["c2", "c3"]:
        for p in pairs:
            a = maps[f"{m}__{p['pair_id']}__mean"]
            base = dict(
                model=m,
                pair_id=p["pair_id"],
                uniprot=p["uniprot"],
                full_inchikey=p["full_inchikey"],
                family=p["family"],
                role="allosteric" if p["binary_label"] else "orthosteric",
            )
            rows.append(
                dict(
                    base,
                    mask_scope="fpocket_union",
                    **metrics(a, np.array(p["masks"]["fpocket_union"], bool)),
                )
            )
            if p["pair_id"] in site:
                rows.append(
                    dict(
                        base,
                        mask_scope="role_site",
                        site_source=src[p["pair_id"]],
                        **metrics(a, site[p["pair_id"]]),
                    )
                )
    pm = pd.DataFrame(rows)
    pm.to_csv(R / "SITE_METRICS.tsv", sep="\t", index=False)
    pm = pm[pm.status == "evaluated"]
    out = []
    for m in ["c2", "c3"]:
        for mask in ["fpocket_union", "role_site"]:
            f = pm[(pm.model == m) & (pm.mask_scope == mask)]
            for role in ["orthosteric", "allosteric"]:
                g = (
                    f[f.role == role]
                    .groupby("uniprot")
                    .agg(v=("attention_enrichment", "mean"), fam=("family", "first"))
                )
                out.append(
                    dict(
                        set="all_pairs",
                        model=m,
                        mask=mask,
                        role=role,
                        unit="protein",
                        **summary(g.v, g.fam, f"a|{m}|{mask}|{role}", "protein"),
                    )
                )
            if mask != "role_site":
                continue
            # within protein: proteins with an orthosteric BioLiP site and an allosteric AlloBench site
            g = (
                f.groupby(["uniprot", "role"])
                .attention_enrichment.mean()
                .unstack()
                .dropna()
            )
            fam = f.groupby("uniprot").family.first().loc[g.index]
            g.assign(family=fam).reset_index().assign(model=m).to_csv(
                R / f"WITHIN_PROTEIN_{m.upper()}.tsv", sep="\t", index=False
            )
            for role in ["orthosteric", "allosteric"]:
                out.append(
                    dict(
                        set="within_protein",
                        model=m,
                        mask=mask,
                        role=role,
                        unit="protein",
                        **summary(g[role], fam, f"b|{m}|{role}", "protein"),
                    )
                )
            # within ligand: exact ligand with an evaluable site in both roles (means over its proteins)
            per = (
                f.groupby(["full_inchikey", "role", "uniprot"])
                .attention_enrichment.mean()
                .groupby(["full_inchikey", "role"])
                .mean()
                .unstack()
                .dropna()
            )
            per.reset_index().assign(model=m).to_csv(
                R / f"WITHIN_LIGAND_{m.upper()}.tsv", sep="\t", index=False
            )
            for role in ["orthosteric", "allosteric"]:
                out.append(
                    dict(
                        set="within_ligand",
                        model=m,
                        mask=mask,
                        role=role,
                        unit="ligand",
                        **summary(per[role], per.index, f"c|{m}|{role}", "ligand"),
                    )
                )
    t = pd.DataFrame(out)
    t.to_csv(R / "UNIFORM_TESTS.tsv", sep="\t", index=False)
    # does the map follow the ligand: own site under own map minus under other same-protein ligands' maps
    spec = []
    for m in ["c2", "c3"]:
        by = {}
        for p in pairs:
            if p["pair_id"] in site:
                by.setdefault(p["uniprot"], []).append(p)
        for u, grp in by.items():
            for p in grp:
                own = site[p["pair_id"]]
                others = [
                    q
                    for q in grp
                    if q["pair_id"] != p["pair_id"]
                    and not np.array_equal(site[q["pair_id"]], own)
                ]
                if not others:
                    continue
                mine = maps[f"{m}__{p['pair_id']}__mean"][own].sum()
                oth = np.mean(
                    [maps[f"{m}__{q['pair_id']}__mean"][own].sum() for q in others]
                )
                spec.append(
                    dict(
                        model=m,
                        pair_id=p["pair_id"],
                        uniprot=u,
                        family=p["family"],
                        role="allosteric" if p["binary_label"] else "orthosteric",
                        other_ligands=len(others),
                        enrichment_delta=(mine - oth) / own.mean(),
                        log_ratio=float(np.log(mine / oth)),
                    )
                )
    sp = pd.DataFrame(spec)
    sp.to_csv(R / "LIGAND_SPECIFICITY.tsv", sep="\t", index=False)
    ss = []
    for m, f in sp.groupby("model"):
        g = f.groupby("uniprot").agg(v=("log_ratio", "mean"), fam=("family", "first"))
        s = summary(np.exp(g.v), g.fam, f"d|{m}", "protein")
        ss.append(
            dict(
                model=m,
                pairs=len(f),
                metric="own/other ligand attention on own site",
                **s,
            )
        )
    pd.DataFrame(ss).to_csv(R / "LIGAND_SPECIFICITY_SUMMARY.tsv", sep="\t", index=False)
    cov = (
        pm[pm.model == "c2"]
        .groupby(["mask_scope", "role"])
        .agg(pairs=("pair_id", "nunique"), proteins=("uniprot", "nunique"))
        .reset_index()
    )
    cov.to_csv(R / "COVERAGE_SUMMARY.tsv", sep="\t", index=False)
    print(cov.to_string(index=False))
    print(
        t[
            [
                "set",
                "model",
                "mask",
                "role",
                "n",
                "geometric_mean",
                "ci_low",
                "ci_high",
                "median",
                "below_1",
                "wilcoxon_p_vs_1",
                "stars",
                "cluster_level_p_vs_1",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(
        pd.DataFrame(ss)[
            [
                "model",
                "pairs",
                "n",
                "geometric_mean",
                "ci_low",
                "ci_high",
                "wilcoxon_p_vs_1",
                "stars",
            ]
        ].to_string(index=False)
    )

    if args.single_maps:
        single = np.load(args.single_maps, allow_pickle=False)
        rows = []
        by = {}
        for p in pairs:
            by.setdefault(p["uniprot"], []).append(p)
        for model in ["c2", "c3"]:
            for uid, group in by.items():
                if len(group) < 2:
                    continue
                between_ligands = [
                    spearmanr(
                        single[f"{model}__{seed}__{a['pair_id']}"],
                        single[f"{model}__{seed}__{b['pair_id']}"],
                    ).statistic
                    for a, b in itertools.combinations(group, 2)
                    for seed in args.seeds
                ]
                between_seeds = [
                    spearmanr(
                        single[f"{model}__{s1}__{p['pair_id']}"],
                        single[f"{model}__{s2}__{p['pair_id']}"],
                    ).statistic
                    for p in group
                    for s1, s2 in itertools.combinations(args.seeds, 2)
                ]
                rows.append(
                    dict(
                        model=model,
                        uniprot=uid,
                        ligands=len(group),
                        between_ligands=float(np.median(between_ligands)),
                        between_seeds=float(np.median(between_seeds)),
                        family=group[0]["family"],
                    )
                )
        pd.DataFrame(rows).to_csv(R / "MAP_SIMILARITY.tsv", sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pairs", "allosteric-masks", "maps", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--single-maps", type=Path)
    parser.add_argument(
        "--seeds", type=int, nargs=3, default=[20260817, 20260818, 20260819]
    )
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
