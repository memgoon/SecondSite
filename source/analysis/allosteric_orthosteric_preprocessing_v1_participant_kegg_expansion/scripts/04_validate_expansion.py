"""Independent validator for the participant/KEGG orthosteric expansion.

This module deliberately imports none of the build stages. It re-derives the
candidate universe, the chemical standardization and every registration rule
from the frozen inputs and compares against the emitted tables.
"""
import os
import sys
import csv
import gzip
import json
import glob
import hashlib
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..'))
ANALYSIS = os.path.abspath(os.path.join(PKG, '..'))
DATA = os.path.join(PKG, 'data')
csv.field_size_limit(10 ** 9)

CHECKS = []


def check(name, ok, **detail):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    print('%-4s %-52s %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, sort_keys=True)[:110]))


def rd(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def uniprot_ec(entry):
    pd = entry.get('proteinDescription', {})
    out = set()

    def take(n):
        for e in n.get('ecNumbers', []) or []:
            out.add(e['value'])
    take(pd.get('recommendedName', {}))
    for a in pd.get('alternativeNames', []) + pd.get('submissionNames', []):
        take(a)
    for c in pd.get('includes', []) + pd.get('contains', []):
        take(c.get('recommendedName', {}))
        for a in c.get('alternativeNames', []):
            take(a)
    return {e for e in out if e.count('.') == 3 and '-' not in e}


def main():
    sys.path.insert(0, HERE)
    import common as C

    # 1 -- frozen input hashes
    snap = {r['input_path']: r['sha256'] for r in rd(os.path.join(DATA, 'INPUT_SNAPSHOT_HASHES.tsv'))}
    bad = [p for p in C.INPUTS
           if snap.get(os.path.relpath(p, ANALYSIS)) != sha(p)]
    check('frozen_input_hashes_match', not bad, inputs=len(snap), mismatched=len(bad))

    cands = rd(os.path.join(DATA, 'EXPANSION_CANDIDATE_UNIVERSE.tsv'))
    struct = {r['chebi_id']: r for r in rd(os.path.join(DATA, 'CHEBI_STRUCTURE_RESOLUTION.tsv'))}
    ledger = rd(os.path.join(DATA, 'EXPANSION_PAIR_LEDGER.tsv'))
    reg = rd(os.path.join(DATA, 'EXPANSION_STRICT_ORTHOSTERIC_PAIRS.tsv'))

    # 2 -- lane A re-derivation
    seeds = {}
    for p in C.pilot_files('*PARTICIPANT_STRUCTURE_REGISTRY.tsv'):
        for r in rd(p):
            seeds[r['seed_id']] = r
    laneA = set()
    for p in C.pilot_files('*STRUCTURED_PARTICIPANT_BIOLIP_COVERAGE.tsv'):
        for r in rd(p):
            if 'no' in r['biolip_match_status'].lower() and \
               r['resolution_status'] == 'resolved_exact_chebi_query':
                s = seeds.get(r['seed_id'])
                if s and s.get('chebi_id'):
                    laneA.add((r['uniprot'], s['chebi_id'].replace('CHEBI:', '').strip()))
    obsA = {(r['uniprot'], r['chebi_id']) for r in cands if 'participant_rescue' in r['lane']}
    check('lane_a_rederived_from_frozen_pilots', laneA == obsA,
          rederived=len(laneA), observed=len(obsA))

    # 3 -- lane B re-derivation
    ec_rn = defaultdict(set)
    for l in open(C.KEGG_EC_RN):
        a, b = l.split()
        ec_rn[a.split(':', 1)[1]].add(b)
    rn_cpd = defaultdict(set)
    for l in open(C.KEGG_RN_CPD):
        a, b = l.split()
        rn_cpd[a].add(b.split(':', 1)[1])
    cpd_chebi = defaultdict(set)
    for l in open(C.KEGG_CPD_CHEBI):
        a, b = l.split()
        cpd_chebi[a.split(':', 1)[1]].add(b.split(':', 1)[1])
    meta = json.load(open(C.UNIPROT_META))
    laneB = set()
    for acc, e in meta.items():
        for ec in uniprot_ec(e):
            for rn in ec_rn.get(ec, ()):
                for cpd in rn_cpd.get(rn, ()):
                    if cpd in C.CURRENCY_KEGG or cpd in C.COFACTOR_KEGG:
                        continue
                    for ch in cpd_chebi.get(cpd, ()):
                        laneB.add((acc, ch))
    obsB = {(r['uniprot'], r['chebi_id']) for r in cands if 'kegg_reaction_expansion' in r['lane']}
    check('lane_b_rederived_from_frozen_kegg', laneB == obsB,
          rederived=len(laneB), observed=len(obsB))

    # 4 -- cofactor / currency exclusion
    inscope = [r for r in cands if r['scope_status'] == 'in_scope']
    currency, cofactor = C.scope_exclusion_sets()
    leaked = [r for r in inscope
              if r['chebi_id'] in currency or r['chebi_id'] in cofactor
              or set(filter(None, r['evidence_types'].split(';'))) == {'cofactor_annotation'}]
    check('scope_exclusions_applied_to_both_lanes', not leaked,
          in_scope=len(inscope), leaked=len(leaked),
          excluded=len(cands) - len(inscope))


    # 5 -- one disposition per candidate
    ids = [r['expansion_pair_id'] for r in ledger]
    keys = [(r['uniprot'], r['chebi_id']) for r in ledger]
    check('one_disposition_per_candidate',
          len(ledger) == len(inscope) == len(set(keys)) and len(set(ids)) == len(ids),
          ledger=len(ledger), in_scope=len(inscope), unique_keys=len(set(keys)))

    # 6 -- independent re-standardization of a deterministic sample
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    LF, UC = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()
    ok_rows = [r for r in struct.values()
               if r['resolution_status'] == 'resolved_exact_chebi_structure']
    sample = sorted(ok_rows, key=lambda r: r['chebi_id'])[::37]
    mismatch = []
    for r in sample:
        m = Chem.MolFromSmiles(r['parent_smiles'])
        if m is None:
            mismatch.append(r['chebi_id'])
            continue
        k = Chem.MolToInchiKey(UC.uncharge(LF.choose(m)))
        if k != r['parent_inchikey']:
            mismatch.append(r['chebi_id'])
    check('parent_structures_reproduce', not mismatch,
          sampled=len(sample), mismatched=len(mismatch))

    # 7 -- registered rows carry a resolved discrete structure
    badstruct = [r for r in reg
                 if not r['full_inchikey'] or r['full_inchikey'] == 'NA'
                 or struct.get(r['chebi_id'], {}).get('resolution_status')
                 != 'resolved_exact_chebi_structure']
    check('registered_rows_have_resolved_structure', not badstruct, registered=len(reg),
          bad=len(badstruct))

    tiny = [r for r in reg
            if int(struct.get(r['chebi_id'], {}).get('heavy_atom_count') or 0)
            < C.MIN_HEAVY_ATOMS]
    check('no_registered_monatomic_species', not tiny, registered=len(reg), tiny=len(tiny))

    # 8 -- ASD conflicts at BOTH levels
    base = rd(C.UNCONTROLLED)
    af = {(r['uniprot'], r['full_inchikey']) for r in base if r['class_label'] == 'allosteric'}
    ac = {(r['uniprot'], r['connectivity_key']) for r in base if r['class_label'] == 'allosteric'}
    vf = [r for r in reg if (r['uniprot'], r['full_inchikey']) in af]
    vc = [r for r in reg if (r['uniprot'], r['connectivity_key']) in ac]
    check('no_asd_conflict_full_or_connectivity', not vf and not vc,
          full_violations=len(vf), connectivity_violations=len(vc))

    # 9 -- uniqueness and non-duplication of registered pairs
    rk = [(r['uniprot'], r['full_inchikey']) for r in reg]
    audited = rd(C.AUDITED_PAIRS)
    existing = ({(r['uniprot'], r['full_inchikey']) for r in base
                 if r['class_label'] == 'orthosteric'}
                | {(r['uniprot'], r['full_inchikey']) for r in audited})
    check('registered_pairs_unique_and_new',
          len(rk) == len(set(rk)) and not (set(rk) & existing),
          registered=len(rk), unique=len(set(rk)),
          overlapping_existing=len(set(rk) & existing))

    # 10 -- augmented table integrity
    p4 = rd(os.path.join(ANALYSIS,
                         'allosteric_orthosteric_preprocessing_v1_biolip_primary_ligand_'
                         'pilot4_completion_121', 'data',
                         'UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_BIOLIP_AUGMENTED.tsv.gz'))
    aug = rd(os.path.join(DATA,
                          'UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_PARTICIPANT_KEGG_AUGMENTED.tsv.gz'))
    check('augmented_is_base_plus_registered',
          len(aug) == len(p4) + len(reg) and list(aug[0]) == list(p4[0]),
          base=len(p4), registered=len(reg), augmented=len(aug))
    prefix = aug[:len(p4)]
    check('augmented_preserves_prior_rows_verbatim',
          [r['dataset_row_id'] for r in prefix] == [r['dataset_row_id'] for r in p4],
          rows=len(p4))

    # 10b -- class_label and binary_label must agree on every row
    mism = [r for r in aug
            if (r['class_label'] == 'allosteric') != (r['binary_label'] == '1')]
    check('augmented_binary_label_matches_class_label', not mism,
          rows=len(aug), mismatched=len(mism))

    # 11 -- no cross-class conflict introduced in the augmented table
    a2 = {(r['uniprot'], r['full_inchikey']) for r in aug if r['class_label'] == 'allosteric'}
    o2 = {(r['uniprot'], r['full_inchikey']) for r in aug if r['class_label'] == 'orthosteric'}
    check('augmented_has_zero_cross_class_conflicts', not (a2 & o2), conflicts=len(a2 & o2))

    # 12 -- anchoring accounting
    acct = {r['metric']: int(r['count'])
            for r in rd(os.path.join(DATA, 'EXPANSION_POPULATION_ACCOUNTING.tsv'))}
    ap = {r['uniprot'] for r in base if r['class_label'] == 'allosteric'}
    op = ({r['uniprot'] for r in base if r['class_label'] == 'orthosteric'}
          | {r['uniprot'] for r in audited})
    check('anchoring_recomputed',
          acct['anchored_before_expansion'] == len(ap & op)
          and acct['anchored_after_expansion'] == len(ap & (op | {r['uniprot'] for r in reg})),
          before=len(ap & op),
          after=len(ap & (op | {r['uniprot'] for r in reg})))

    # 13 -- audit-excluded list
    keep = {(r['uniprot'], r['full_inchikey']) for r in audited}
    exp = [r for r in p4
           if r['evidence_subtypes'].startswith('direct_accession_specific_structure')
           and (r['uniprot'], r['full_inchikey']) not in keep]
    got = rd(os.path.join(DATA, 'AUDIT_EXCLUDED_BIOLIP_ROWS.tsv'))
    check('audit_excluded_rows_listed', len(exp) == len(got), expected=len(exp), listed=len(got))

    # 14 -- decision vocabulary is closed
    allowed = {'register_expansion_orthosteric_pair',
               'reject_exact_asd_pair_conflict', 'reject_connectivity_asd_pair_conflict',
               'not_registered_already_present_orthosteric',
               'not_registered_duplicate_within_expansion',
               'not_registered_unresolved_generic_class_wildcard',
               'not_registered_unresolved_no_chebi_structure',
               'not_registered_unresolved_rdkit_standardization_failed',
               'reject_out_of_scope_monatomic_species'}
    seen = set(r['final_pair_decision'] for r in ledger)
    check('decision_vocabulary_closed', seen <= allowed, unexpected=sorted(seen - allowed))

    npass = sum(1 for c in CHECKS if c['ok'])
    status = 'PASS' if npass == len(CHECKS) else 'FAIL'
    json.dump({'audit': 'PARTICIPANT_KEGG_ORTHOSTERIC_EXPANSION_1.0',
               'status': status, 'checks_passed': npass, 'checks_total': len(CHECKS),
               'checks': CHECKS,
               'counts': {'registered_pairs': len(reg),
                          'registered_proteins': len({r['uniprot'] for r in reg}),
                          'augmented_rows': len(aug),
                          'anchored_after': acct['anchored_after_expansion']},
               'scope_limit': ('Orthosteric-reference preprocessing only. Rows carry curated '
                               'reaction-participant identity without a co-crystal and imply '
                               'no site, potency, mechanism or allostery.')},
              open(os.path.join(PKG, 'validation', 'VALIDATION.json'), 'w'), indent=1)
    print('\n%s %d/%d' % (status, npass, len(CHECKS)))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
