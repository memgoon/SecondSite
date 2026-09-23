"""Shared frozen paths and helpers for the participant/KEGG orthosteric expansion.

The package is append-only. Every input listed in ``INPUTS`` is read-only and is
hashed into ``data/INPUT_SNAPSHOT_HASHES.tsv`` before any table is written.
"""
import csv
import glob
import gzip
import hashlib
import os

csv.field_size_limit(10 ** 9)

ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(PKG, 'data')
CACHE = os.path.join(PKG, 'cache')

BASE = os.path.join(ANALYSIS, 'allosteric_orthosteric_preprocessing_v1')
AUDIT = os.path.join(
    ANALYSIS, 'allosteric_orthosteric_preprocessing_v1_strict_pair_multisite_audit')
PILOT_GLOB = os.path.join(
    ANALYSIS, 'allosteric_orthosteric_preprocessing_v1_biolip_primary_ligand_*')

UNCONTROLLED = os.path.join(BASE, 'data', 'UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC.tsv.gz')
AUDITED_PAIRS = os.path.join(AUDIT, 'data', 'FILTERED_STRICT_ORTHOSTERIC_PAIRS.tsv')

KEGG_EC_RN = os.path.join(CACHE, 'kegg', 'kegg_ec_rn.tsv')
KEGG_RN_CPD = os.path.join(CACHE, 'kegg', 'kegg_rn_cpd.tsv')
KEGG_CPD_CHEBI = os.path.join(CACHE, 'kegg', 'kegg_cpd_chebi.tsv')
CHEBI_STRUCTURES = os.path.join(CACHE, 'chebi', 'structures.tsv.gz')
CHEBI_COMPOUNDS = os.path.join(CACHE, 'chebi', 'compounds.tsv.gz')
CHEBI_RELATIONS = os.path.join(CACHE, 'chebi', 'relation.tsv.gz')
UNIPROT_META = os.path.join(CACHE, 'meta635.json')


def pilot_files(pattern):
    return sorted(glob.glob(os.path.join(PILOT_GLOB, 'data', pattern)))


INPUTS = ([UNCONTROLLED, AUDITED_PAIRS, KEGG_EC_RN, KEGG_RN_CPD, KEGG_CPD_CHEBI,
           CHEBI_STRUCTURES, CHEBI_COMPOUNDS, CHEBI_RELATIONS, UNIPROT_META]
          + pilot_files('*STRUCTURED_PARTICIPANT_BIOLIP_COVERAGE.tsv')
          + pilot_files('*PARTICIPANT_STRUCTURE_REGISTRY.tsv'))

# Canonical cofactors, redox carriers and catalytic/structural metals.
# Excluded from the orthosteric pair scope by contract, matching the frozen
# BioLiP pilot policy (`reject_out_of_pair_scope_cofactor`).
COFACTOR_KEGG = {
    'C00003', 'C00004', 'C00005', 'C00006',          # NAD(H), NADP(H)
    'C00016', 'C01352',                              # FAD, FADH2
    'C00061', 'C01847',                              # FMN, FMNH2
    'C00010', 'C00024', 'C00083', 'C00332',          # CoA and common acyl-CoAs
    'C00019', 'C00021',                              # SAM, SAH
    'C00018', 'C00647',                              # PLP, PMP
    'C00101', 'C00415', 'C00504',                    # THF, DHF, folate
    'C00120', 'C00068', 'C00378',                    # biotin, TPP, thiamine
    'C00032', 'C00034', 'C00038', 'C14818', 'C14819',
    'C00305', 'C00076', 'C01330', 'C00238', 'C00291',
    'C00070', 'C00175', 'C00051', 'C00127',
    'C00194', 'C00272', 'C00268',
}

# Ubiquitous reaction currency that does not define a protein-ligand pair.
CURRENCY_KEGG = {
    'C00001', 'C00080', 'C00007', 'C00011', 'C00014',
    'C00013', 'C00288', 'C00027', 'C05359', 'C00205',
}

# Scope is a property of the chemical, not of the lane it arrived through, so
# both lanes are filtered by the same ChEBI-level sets. The KEGG identifiers
# above are projected onto ChEBI through the cached conversion table and are
# supplemented with entries that reach the participant lane directly from
# UniProt catalytic-reaction annotation.
CURRENCY_CHEBI_EXTRA = {
    '15377', '15378', '16234', '29412',            # water, proton, hydroxide, oxonium
    '15379', '25805', '26689',                     # dioxygen, oxygen atom, oxygen species
    '16526', '13282',                              # carbon dioxide
    '16134', '28938', '29337',                     # ammonia / ammonium
    '33019', '18361', '29888',                     # diphosphate species
    '17544', '16995',                              # hydrogencarbonate, oxalate carrier forms
    '16240', '15379',                              # hydrogen peroxide
    '10545', '30212',                              # electron, photon
}

# A discrete ligand needs more than a single heavy atom. Bare metal ions,
# halides and monatomic chalcogens are catalytic or structural cofactors under
# the frozen pilot policy and are never registered as pair-defining chemistry.
MIN_HEAVY_ATOMS = 2


# ChEBI relation types used to close a scope set over protonation states.
# 6/7 are conjugate acid/base, 11 is tautomer. `has_functional_parent` is
# deliberately NOT followed: acetyl-CoA has acetic acid as a functional parent,
# so following it would drag genuine substrates into the cofactor set.
_CONJUGATE_RELATIONS = {'6', '7', '11'}
_CLOSURE_DEPTH = 2


def _chebi_conjugate_graph():
    acc = {}
    with gzip.open(CHEBI_COMPOUNDS, 'rt', encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            a = (r.get('chebi_accession') or '').replace('CHEBI:', '').strip()
            if a:
                acc[r['id']] = a
    graph = {}
    with gzip.open(CHEBI_RELATIONS, 'rt', encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['relation_type_id'] not in _CONJUGATE_RELATIONS:
                continue
            i, f = acc.get(r['init_id']), acc.get(r['final_id'])
            if i and f:
                graph.setdefault(i, set()).add(f)
                graph.setdefault(f, set()).add(i)
    return graph


def _close(seed, graph, depth=_CLOSURE_DEPTH):
    out = set(seed)
    frontier = [(s, 0) for s in seed]
    while frontier:
        node, d = frontier.pop()
        if d >= depth:
            continue
        for nxt in graph.get(node, ()):  # noqa: E501
            if nxt not in out:
                out.add(nxt)
                frontier.append((nxt, d + 1))
    return out


def scope_exclusion_sets():
    """ChEBI identifier sets for reaction currency and cofactors.

    The KEGG-derived lists reproduce the frozen BioLiP pilot policy. UniProt
    annotates participants in charged forms (NAD(1-), coenzyme A(4-)) that carry
    different ChEBI identifiers from the neutral entries, so each set is closed
    over conjugate acid/base and tautomer relations. Only those relations are
    followed, which keeps ATP, ADP and acyl-CoA thioesters - registered as
    substrates by the accepted pilots - out of the cofactor set.
    """
    cpd_chebi = {}
    with open(KEGG_CPD_CHEBI) as fh:
        for line in fh:
            a, b = line.split()
            cpd_chebi.setdefault(a.split(':', 1)[1], set()).add(b.split(':', 1)[1])
    currency = set(CURRENCY_CHEBI_EXTRA)
    for cpd in CURRENCY_KEGG:
        currency |= cpd_chebi.get(cpd, set())
    cofactor = set()
    for cpd in COFACTOR_KEGG:
        cofactor |= cpd_chebi.get(cpd, set())
    graph = _chebi_conjugate_graph()
    currency = _close(currency, graph)
    cofactor = _close(cofactor, graph)
    return currency, cofactor - currency


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_tsv(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def write_tsv(path, rows, fields):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    os.replace(tmp, path)


def uniprot_ec(entry):
    """Exact four-level EC numbers declared on a UniProt entry."""
    pd = entry.get('proteinDescription', {})
    out = set()

    def take(node):
        for n in node.get('ecNumbers', []) or []:
            out.add(n['value'])

    take(pd.get('recommendedName', {}))
    for alt in pd.get('alternativeNames', []) + pd.get('submissionNames', []):
        take(alt)
    for comp in pd.get('includes', []) + pd.get('contains', []):
        take(comp.get('recommendedName', {}))
        for alt in comp.get('alternativeNames', []):
            take(alt)
    return {e for e in out if e.count('.') == 3 and '-' not in e}
