"""Stage 3 - register the expansion pairs and emit the augmented pair universe.

A candidate becomes a registered orthosteric row when all of the following hold:

* the accession owns the annotation exactly (UniProt accession-level ChEBI
  participant, or a KEGG reaction attached to an exact four-level EC declared on
  that accession);
* the ChEBI identifier resolves to a discrete standardized parent structure;
* the compound is not a cofactor, redox carrier, metal or reaction currency,
  and carries more than one heavy atom, so bare ions are never pair-defining;
* the pair does not conflict with an ASD allosteric record for the same
  accession, tested at BOTH the full InChIKey and the connectivity-key level.

The connectivity-level test is deliberately stricter than the frozen BioLiP
pilots, which tested only the full InChIKey while accepting standardized-parent
chemical identity. That asymmetry is closed here.

These rows carry curated reaction-participant evidence WITHOUT a co-crystal of
that accession with that compound, and are tiered accordingly. They are not
site-level observations and no potency, kinetic mechanism or allostery is implied.
"""
import io
import os
import re
import sys
import gzip
import csv
import hashlib
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PAIR_FIELDS = ['expansion_pair_id', 'uniprot', 'chebi_id', 'chebi_name', 'lane',
               'evidence_types', 'reported_name', 'kegg_compound', 'kegg_reactions',
               'kegg_ecs', 'parent_smiles', 'full_inchikey', 'connectivity_key',
               'structure_route', 'parent_is_neutral', 'cofactor_scaffold_flag',
               'evidence_tier', 'final_pair_decision', 'class_label', 'claim_limit']

# Advisory only. Some scaffolds that usually act as cofactors are genuine
# substrates for particular enzymes - dihydrofolate for dihydrofolate reductase,
# acyl-CoA thioesters for acyltransferases - so these are flagged for downstream
# review rather than excluded. Exclusion is governed solely by the ChEBI scope
# sets in `common.scope_exclusion_sets`.
SCAFFOLD_RE = re.compile(
    r'\b(coenzyme a|CoA|NAD|NADP|FAD|FMN|flavin|thiamine|thiamin|pyridox|'
    r'folate|folic|pteroyl|tetrahydrofolate|dihydrofolate|biotin|cobalamin|'
    r'corrin|haem|heme|porphyrin|adenosylmethionine|adenosylhomocysteine|'
    r'glutathione|ubiquinone|menaquinone|lipoate|lipoamide|molybdopterin)\b',
    re.I)

TIER = 'curated_reaction_participant_without_cocrystal'
LIMIT = ('Curated accession-level reaction-participant identity only; no co-crystal '
         'of this accession with this compound was required or implied, and no site '
         'assignment, potency, kinetic mechanism or allostery is inferred.')


def row_id(prefix, *parts):
    return prefix + hashlib.sha256('|'.join(parts).encode()).hexdigest()[:20]


def chebi_names():
    names = {}
    with gzip.open(C.CHEBI_COMPOUNDS, 'rt', encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            a = (r.get('chebi_accession') or '').replace('CHEBI:', '').strip()
            if a and r.get('name'):
                names[a] = re.sub(r'<[^>]+>', '', r['name'])
    return names


def main():
    cands = [r for r in C.read_tsv(os.path.join(C.DATA, 'EXPANSION_CANDIDATE_UNIVERSE.tsv'))
             if r['scope_status'] == 'in_scope']
    names = chebi_names()
    struct = {r['chebi_id']: r for r in
              C.read_tsv(os.path.join(C.DATA, 'CHEBI_STRUCTURE_RESOLUTION.tsv'))}

    base = C.read_tsv(C.UNCONTROLLED)
    allo_full = {(r['uniprot'], r['full_inchikey']) for r in base
                 if r['class_label'] == 'allosteric'}
    allo_conn = {(r['uniprot'], r['connectivity_key']) for r in base
                 if r['class_label'] == 'allosteric'}
    orth_full = {(r['uniprot'], r['full_inchikey']) for r in base
                 if r['class_label'] == 'orthosteric'}
    audited = C.read_tsv(C.AUDITED_PAIRS)
    orth_full |= {(r['uniprot'], r['full_inchikey']) for r in audited}

    rows, seen = [], set()
    for c in sorted(cands, key=lambda r: (r['uniprot'], r['chebi_id'])):
        s = struct.get(c['chebi_id'])
        out = dict(c)
        out['expansion_pair_id'] = row_id('PKEP', c['uniprot'], c['chebi_id'])
        out['class_label'] = 'orthosteric'
        out['evidence_tier'] = TIER
        out['claim_limit'] = LIMIT
        out['chebi_name'] = names.get(c['chebi_id'], 'NA')
        out['cofactor_scaffold_flag'] = ('true' if SCAFFOLD_RE.search(out['chebi_name'])
                                         else 'false')
        if not s or s['resolution_status'] != 'resolved_exact_chebi_structure':
            out['final_pair_decision'] = 'not_registered_' + (
                s['resolution_status'] if s else 'unresolved_no_chebi_structure')
            out.update({'parent_smiles': 'NA', 'full_inchikey': 'NA',
                        'connectivity_key': 'NA', 'structure_route': 'NA',
                        'parent_is_neutral': 'NA'})
            rows.append(out)
            continue
        try:
            heavy = int(s['heavy_atom_count'])
        except (TypeError, ValueError):
            heavy = 0
        if heavy < C.MIN_HEAVY_ATOMS:
            out['final_pair_decision'] = 'reject_out_of_scope_monatomic_species'
            out.update({'parent_smiles': s['parent_smiles'],
                        'full_inchikey': s['parent_inchikey'],
                        'connectivity_key': s['connectivity_key'],
                        'structure_route': s['structure_route'],
                        'parent_is_neutral': s['parent_is_neutral']})
            rows.append(out)
            continue
        key = (c['uniprot'], s['parent_inchikey'])
        conn = (c['uniprot'], s['connectivity_key'])
        out.update({'parent_smiles': s['parent_smiles'],
                    'full_inchikey': s['parent_inchikey'],
                    'connectivity_key': s['connectivity_key'],
                    'structure_route': s['structure_route'],
                    'parent_is_neutral': s['parent_is_neutral']})
        if key in allo_full:
            out['final_pair_decision'] = 'reject_exact_asd_pair_conflict'
        elif conn in allo_conn:
            out['final_pair_decision'] = 'reject_connectivity_asd_pair_conflict'
        elif key in orth_full:
            out['final_pair_decision'] = 'not_registered_already_present_orthosteric'
        elif key in seen:
            out['final_pair_decision'] = 'not_registered_duplicate_within_expansion'
        else:
            seen.add(key)
            out['final_pair_decision'] = 'register_expansion_orthosteric_pair'
        rows.append(out)
    C.write_tsv(os.path.join(C.DATA, 'EXPANSION_PAIR_LEDGER.tsv'), rows, PAIR_FIELDS)

    reg = [r for r in rows if r['final_pair_decision'] == 'register_expansion_orthosteric_pair']
    C.write_tsv(os.path.join(C.DATA, 'EXPANSION_STRICT_ORTHOSTERIC_PAIRS.tsv'),
                reg, PAIR_FIELDS)

    # ---- augmented uncontrolled universe -------------------------------------
    base_aug = C.read_tsv(os.path.join(
        C.ANALYSIS,
        'allosteric_orthosteric_preprocessing_v1_biolip_primary_ligand_pilot4_completion_121',
        'data', 'UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_BIOLIP_AUGMENTED.tsv.gz'))
    fields = list(base_aug[0])
    new = []
    for r in reg:
        src = ['UniProt', 'ChEBI'] + (['KEGG'] if r['kegg_compound'] != 'NA' else [])
        lanes = r['lane'].split(';')
        new.append({
            'dataset_row_id': row_id('UNCONTROLLED_PKE_', r['uniprot'], r['full_inchikey']),
            'uniprot': r['uniprot'], 'full_inchikey': r['full_inchikey'],
            'connectivity_key': r['connectivity_key'],
            'canonical_smiles': r['parent_smiles'], 'fingerprint_eligible': 'true',
            # binary_label is the allosteric indicator: 1 allosteric, 0 orthosteric
            'class_label': 'orthosteric', 'binary_label': '0',
            'source_databases': ';'.join(src), 'source_lanes': ';'.join(lanes),
            'source_pair_ids': r['expansion_pair_id'],
            'evidence_subtypes': TIER,
            'n_source_pair_rows': '1', 'n_evidence_rows': '1',
            'protein_control_status': 'uncontrolled_all_proteins',
            'ligand_control_status': 'uncontrolled_no_property_grouping',
            'class_conflict_status': 'nonconflicting', 'decoy_included': 'false',
            'claim_limit': LIMIT})
    aug = base_aug + new
    out = os.path.join(C.DATA,
                       'UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_PARTICIPANT_KEGG_AUGMENTED.tsv.gz')
    tmp = out + '.tmp'
    # mtime=0 so the gzip container is byte-reproducible across runs
    with open(tmp, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=fields, delimiter='\t',
                                   lineterminator='\n')
                w.writeheader()
                for r in aug:
                    w.writerow({k: r.get(k, '') for k in fields})
    os.replace(tmp, out)

    # rows the multisite audit removed from the BioLiP set, listed so a clean
    # variant can be produced without modifying any prior package
    audit_keep = {(r['uniprot'], r['full_inchikey']) for r in audited}
    excl = [{'dataset_row_id': r['dataset_row_id'], 'uniprot': r['uniprot'],
             'full_inchikey': r['full_inchikey'],
             'reason': 'removed_by_strict_pair_multisite_audit'}
            for r in base_aug
            if r['evidence_subtypes'].startswith('direct_accession_specific_structure')
            and (r['uniprot'], r['full_inchikey']) not in audit_keep]
    C.write_tsv(os.path.join(C.DATA, 'AUDIT_EXCLUDED_BIOLIP_ROWS.tsv'), excl,
                ['dataset_row_id', 'uniprot', 'full_inchikey', 'reason'])

    # ---- accounting ----------------------------------------------------------
    allo_prot = {r['uniprot'] for r in base if r['class_label'] == 'allosteric'}
    orth_prot = ({r['uniprot'] for r in base if r['class_label'] == 'orthosteric'}
                 | {r['uniprot'] for r in audited})
    acc = [('allosteric_proteins', len(allo_prot)),
           ('anchored_before_expansion', len(allo_prot & orth_prot)),
           ('anchored_after_expansion', len(allo_prot & (orth_prot | {r['uniprot'] for r in reg}))),
           ('expansion_candidates_in_scope', len(cands)),
           ('registered_expansion_pairs', len(reg)),
           ('registered_expansion_proteins', len({r['uniprot'] for r in reg})),
           ('augmented_table_rows', len(aug)),
           ('audit_excluded_biolip_rows', len(excl))]
    for k, v in Counter(r['final_pair_decision'] for r in rows).most_common():
        acc.append(('decision_' + k, v))
    C.write_tsv(os.path.join(C.DATA, 'EXPANSION_POPULATION_ACCOUNTING.tsv'),
                [{'metric': k, 'count': v} for k, v in acc], ['metric', 'count'])

    flagged = [r for r in reg if r['cofactor_scaffold_flag'] == 'true']
    print('PASS registered=%d proteins=%d augmented_rows=%d scaffold_flagged=%d'
          % (len(reg), len({r['uniprot'] for r in reg}), len(aug), len(flagged)))
    for k, v in acc:
        print('   %-52s %6d' % (k, v))


if __name__ == '__main__':
    main()
