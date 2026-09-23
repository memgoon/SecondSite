"""Stage 1 - assemble the protein x ChEBI candidate universe for both lanes.

Lane A (participant_rescue)
    UniProt accession-level catalytic-reaction / binding-feature participants
    that the frozen BioLiP pilots extracted but could not register because no
    co-crystal of that accession with that compound exists. The protein itself
    is structurally characterised; only the specific complex is missing.

Lane B (kegg_reaction_expansion)
    Compounds of the KEGG reactions attached to the exact four-level EC numbers
    declared on the accession.

Cofactors, redox carriers, catalytic/structural metals and reaction currency
are removed by contract in both lanes. No outcome or model information is used.
"""
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FIELDS = ['candidate_id', 'uniprot', 'chebi_id', 'lane', 'evidence_types',
          'reported_name', 'kegg_compound', 'kegg_reactions', 'kegg_ecs',
          'pilot_seed_ids', 'pilot_parent_inchikey', 'scope_status']


def lane_a():
    """Participant seeds with no BioLiP co-crystal for that exact protein-compound."""
    seeds = {}
    for path in C.pilot_files('*PARTICIPANT_STRUCTURE_REGISTRY.tsv'):
        for r in C.read_tsv(path):
            seeds[r['seed_id']] = r
    out = defaultdict(lambda: {'seed_ids': set(), 'ev': set(), 'names': set(),
                               'parent': set()})
    for path in C.pilot_files('*STRUCTURED_PARTICIPANT_BIOLIP_COVERAGE.tsv'):
        for r in C.read_tsv(path):
            if 'no' not in r['biolip_match_status'].lower():
                continue
            if r['resolution_status'] != 'resolved_exact_chebi_query':
                continue
            s = seeds.get(r['seed_id'])
            if not s or not s.get('chebi_id'):
                continue
            chebi = s['chebi_id'].replace('CHEBI:', '').strip()
            if not chebi:
                continue
            rec = out[(r['uniprot'], chebi)]
            rec['seed_ids'].add(r['seed_id'])
            rec['ev'] |= {e for e in (s.get('evidence_types') or '').split(';') if e}
            if (s.get('reported_names') or 'NA') != 'NA':
                rec['names'].add(s['reported_names'])
            if s.get('reference_parent_inchikey'):
                rec['parent'].add(s['reference_parent_inchikey'])
    return out


def lane_b():
    """KEGG reaction compounds reachable from the accession's exact EC numbers."""
    ec_rn = defaultdict(set)
    for line in open(C.KEGG_EC_RN):
        a, b = line.split()
        ec_rn[a.split(':', 1)[1]].add(b)
    rn_cpd = defaultdict(set)
    for line in open(C.KEGG_RN_CPD):
        a, b = line.split()
        rn_cpd[a].add(b.split(':', 1)[1])
    cpd_chebi = defaultdict(set)
    for line in open(C.KEGG_CPD_CHEBI):
        a, b = line.split()
        cpd_chebi[a.split(':', 1)[1]].add(b.split(':', 1)[1])

    meta = json.load(open(C.UNIPROT_META))
    out = defaultdict(lambda: {'cpd': set(), 'rn': set(), 'ec': set()})
    for acc, entry in meta.items():
        for ec in C.uniprot_ec(entry):
            for rn in ec_rn.get(ec, ()):
                for cpd in rn_cpd.get(rn, ()):
                    if cpd in C.CURRENCY_KEGG or cpd in C.COFACTOR_KEGG:
                        continue
                    for chebi in cpd_chebi.get(cpd, ()):
                        rec = out[(acc, chebi)]
                        rec['cpd'].add(cpd)
                        rec['rn'].add(rn.replace('rn:', ''))
                        rec['ec'].add(ec)
    return out


def main():
    a, b = lane_a(), lane_b()
    currency, cofactor = C.scope_exclusion_sets()
    rows, n = [], 0
    for key in sorted(set(a) | set(b)):
        acc, chebi = key
        ra, rb = a.get(key), b.get(key)
        ev = sorted(ra['ev']) if ra else []
        # Scope is a property of the chemical and is applied identically to both
        # lanes. Reaction currency and cofactors are out of pair scope however
        # the compound was annotated; a seed whose only UniProt evidence is a
        # cofactor annotation is additionally out of scope, as in the pilots.
        if chebi in currency:
            scope = 'out_of_scope_reaction_currency'
        elif chebi in cofactor:
            scope = 'out_of_scope_cofactor'
        elif bool(ev) and set(ev) == {'cofactor_annotation'}:
            scope = 'out_of_scope_cofactor_only_evidence'
        else:
            scope = 'in_scope' 
        n += 1
        rows.append({
            'candidate_id': 'PKE%05d' % n,
            'uniprot': acc,
            'chebi_id': chebi,
            'lane': ('participant_rescue;kegg_reaction_expansion' if ra and rb
                     else 'participant_rescue' if ra else 'kegg_reaction_expansion'),
            'evidence_types': ';'.join(ev),
            'reported_name': ';'.join(sorted(ra['names'])) if ra and ra['names'] else 'NA',
            'kegg_compound': ';'.join(sorted(rb['cpd'])) if rb else 'NA',
            'kegg_reactions': ';'.join(sorted(rb['rn'])) if rb else 'NA',
            'kegg_ecs': ';'.join(sorted(rb['ec'])) if rb else 'NA',
            'pilot_seed_ids': ';'.join(sorted(ra['seed_ids'])) if ra else 'NA',
            'pilot_parent_inchikey': (';'.join(sorted(ra['parent']))
                                      if ra and ra['parent'] else 'NA'),
            'scope_status': scope,
        })
    C.write_tsv(os.path.join(C.DATA, 'EXPANSION_CANDIDATE_UNIVERSE.tsv'), rows, FIELDS)

    hashes = [{'input_path': os.path.relpath(p, C.ANALYSIS), 'sha256': C.sha256(p)}
              for p in C.INPUTS]
    C.write_tsv(os.path.join(C.DATA, 'INPUT_SNAPSHOT_HASHES.tsv'), hashes,
                ['input_path', 'sha256'])

    ins = [r for r in rows if r['scope_status'] == 'in_scope']
    from collections import Counter
    print('PASS candidates=%d in_scope=%d proteins=%d chebi=%d'
          % (len(rows), len(ins), len({r['uniprot'] for r in ins}),
             len({r['chebi_id'] for r in ins})))
    for k, v in Counter(r['scope_status'] for r in rows).most_common():
        print('   %-38s %5d' % (k, v))
    for lane in ('participant_rescue', 'kegg_reaction_expansion',
                 'participant_rescue;kegg_reaction_expansion'):
        sub = [r for r in ins if r['lane'] == lane]
        print('  %-45s %5d pairs / %3d proteins'
              % (lane, len(sub), len({r['uniprot'] for r in sub})))


if __name__ == '__main__':
    main()
