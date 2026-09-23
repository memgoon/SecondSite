"""Stage 5 - coverage and per-protein depth of the augmented pair universe.

Depth is reported as the minority-class row count per anchored protein,
min(allosteric rows, orthosteric rows), because that is what bounds any
within-protein contrast an evaluation split can build.
"""
import os
import sys
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FIELDS = ['stage', 'anchored_proteins', 'unanchored_proteins',
          'median_total_rows', 'mean_total_rows',
          'median_minority', 'mean_minority', 'proteins_minority_ge2',
          'proteins_minority_ge3', 'proteins_minority_ge5', 'orthosteric_rows_added']


def profile(stage, allo, orth, added):
    """Two complementary views of per-protein depth.

    `total_rows` is how much labelled data a protein contributes overall.
    `minority` is min(allosteric, orthosteric): the number of matched
    within-protein contrasts an evaluation split can actually build, which is
    capped by whichever class is thinner.
    """
    anchored = [u for u in allo if u in orth]
    mino = [min(len(allo[u]), len(orth[u])) for u in anchored]
    tot = [len(allo[u]) + len(orth[u]) for u in anchored]
    return {'stage': stage, 'anchored_proteins': len(anchored),
            'unanchored_proteins': len(allo) - len(anchored),
            'median_total_rows': '%.0f' % statistics.median(tot),
            'mean_total_rows': '%.2f' % (sum(tot) / len(tot)),
            'median_minority': '%.0f' % statistics.median(mino),
            'mean_minority': '%.2f' % (sum(mino) / len(mino)),
            'proteins_minority_ge2': sum(1 for m in mino if m >= 2),
            'proteins_minority_ge3': sum(1 for m in mino if m >= 3),
            'proteins_minority_ge5': sum(1 for m in mino if m >= 5),
            'orthosteric_rows_added': added}


def main():
    base = C.read_tsv(C.UNCONTROLLED)
    allo, orth = defaultdict(set), defaultdict(set)
    for r in base:
        (allo if r['class_label'] == 'allosteric' else orth)[r['uniprot']].add(r['full_inchikey'])
    rows = [profile('base_uncontrolled', allo, dict(orth), 0)]

    audited = C.read_tsv(C.AUDITED_PAIRS)
    o2 = defaultdict(set, {k: set(v) for k, v in orth.items()})
    for r in audited:
        o2[r['uniprot']].add(r['full_inchikey'])
    rows.append(profile('plus_audited_biolip_pairs', allo, o2, len(audited)))

    reg = C.read_tsv(os.path.join(C.DATA, 'EXPANSION_STRICT_ORTHOSTERIC_PAIRS.tsv'))
    o3 = defaultdict(set, {k: set(v) for k, v in o2.items()})
    for r in reg:
        o3[r['uniprot']].add(r['full_inchikey'])
    rows.append(profile('plus_participant_kegg_expansion', allo, o3, len(reg)))

    C.write_tsv(os.path.join(C.DATA, 'EXPANSION_DEPTH_METRICS.tsv'), rows, FIELDS)

    # the allosteric side is what caps min(); report it so the bound is explicit
    n = [len(v) for v in allo.values()]
    ceil = [{'metric': 'allosteric_proteins', 'count': len(n)},
            {'metric': 'allosteric_ligands_median', 'count': int(statistics.median(n))},
            {'metric': 'proteins_with_exactly_one_allosteric_ligand',
             'count': sum(1 for x in n if x == 1)},
            {'metric': 'ceiling_proteins_minority_ge2', 'count': sum(1 for x in n if x >= 2)},
            {'metric': 'ceiling_proteins_minority_ge3', 'count': sum(1 for x in n if x >= 3)},
            {'metric': 'ceiling_proteins_minority_ge5', 'count': sum(1 for x in n if x >= 5)}]
    C.write_tsv(os.path.join(C.DATA, 'ALLOSTERIC_SIDE_DEPTH_CEILING.tsv'), ceil,
                ['metric', 'count'])

    for r in rows:
        print('%-32s anchored=%4s | total rows med=%2s mean=%5s | minority med=%s mean=%s >=2 %3s >=3 %3s >=5 %3s'
              % (r['stage'], r['anchored_proteins'], r['median_total_rows'],
                 r['mean_total_rows'], r['median_minority'], r['mean_minority'],
                 r['proteins_minority_ge2'], r['proteins_minority_ge3'],
                 r['proteins_minority_ge5']))
    print()
    for c in ceil:
        print('   %-46s %d' % (c['metric'], c['count']))


if __name__ == '__main__':
    main()
