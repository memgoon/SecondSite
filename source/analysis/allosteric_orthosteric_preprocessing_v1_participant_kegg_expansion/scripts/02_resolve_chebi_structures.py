"""Stage 2 - resolve every candidate ChEBI identifier to a standardized parent.

Structures come from the ChEBI flat-file release cached in ``cache/chebi``; no
name lookup or fuzzy matching is used. Standardization repeats the protocol the
frozen BioLiP pilots applied: sanitize, keep the principal (largest) fragment,
neutralize, then emit canonical isomeric parent SMILES and the parent InChIKey.

ChEBI stores some heteroaromatic tautomers in a form RDKit cannot kekulize from
SMILES. For those the cached ``standard_inchi`` is used instead; the route taken
is recorded per row so the choice is auditable rather than silent.
"""
import os
import sys
import csv
import gzip
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

FIELDS = ['chebi_id', 'chebi_internal_id', 'resolution_status', 'structure_route',
          'source_smiles', 'source_inchikey', 'parent_smiles', 'parent_inchikey',
          'connectivity_key', 'parent_is_neutral', 'heavy_atom_count']

_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def _finish(mol):
    mol = _LARGEST.choose(mol)
    if mol is None:
        return None, None, 0
    mol = _UNCHARGER.uncharge(mol)
    if mol is None:
        return None, None, 0
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None, 0
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    key = Chem.MolToInchiKey(mol)
    return (smi or None), (key or None), mol.GetNumHeavyAtoms()


def standardize(smiles, standard_inchi):
    """Return (parent_smiles, parent_inchikey, route, heavy_atom_count)."""
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            smi, key, ha = _finish(mol)
            if key:
                return smi, key, 'chebi_smiles', ha
    if standard_inchi:
        mol = Chem.MolFromInchi(standard_inchi)
        if mol is not None:
            smi, key, ha = _finish(mol)
            if key:
                return smi, key, 'chebi_standard_inchi', ha
    return None, None, 'none', 0


def load_compound_index(needed):
    """chebi accession -> internal compound id, following merged parents."""
    acc2id, parent = {}, {}
    with gzip.open(os.path.join(C.CACHE, 'chebi', 'compounds.tsv.gz'), 'rt') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            acc = (r.get('chebi_accession') or '').replace('CHEBI:', '').strip()
            if acc:
                acc2id[acc] = r['id']
            if r.get('parent_id'):
                parent[r['id']] = r['parent_id']
    resolved = {}
    for chebi in needed:
        cid = acc2id.get(chebi)
        seen = set()
        while cid and cid in parent and cid not in seen:
            seen.add(cid)
            cid = parent[cid]
        if cid:
            resolved[chebi] = cid
    return resolved


def load_structures(internal_ids):
    want = set(internal_ids)
    best = {}
    with gzip.open(C.CHEBI_STRUCTURES, 'rt') as fh:
        rd = csv.reader(fh, delimiter='\t')
        head = next(rd)
        ix = {c: i for i, c in enumerate(head)}
        for row in rd:
            if len(row) != len(head):
                continue
            cid = row[ix['compound_id']]
            if cid not in want:
                continue
            smi = row[ix['smiles']].strip()
            ich = row[ix['standard_inchi']].strip()
            if not smi and not ich:
                continue
            default = row[ix['default_structure']].strip().lower() == 'true'
            if cid not in best or (default and not best[cid][3]):
                best[cid] = (smi, ich, row[ix['standard_inchi_key']].strip(), default)
    return best


def main():
    cands = C.read_tsv(os.path.join(C.DATA, 'EXPANSION_CANDIDATE_UNIVERSE.tsv'))
    needed = sorted({r['chebi_id'] for r in cands if r['scope_status'] == 'in_scope'})
    print('chebi identifiers to resolve:', len(needed), flush=True)

    acc2int = load_compound_index(needed)
    struct = load_structures(set(acc2int.values()))
    print('mapped=%d structures=%d' % (len(acc2int), len(struct)), flush=True)

    rows = []
    for chebi in needed:
        cid = acc2int.get(chebi)
        rec = struct.get(cid) if cid else None
        base = {'chebi_id': chebi, 'chebi_internal_id': cid or 'NA',
                'structure_route': 'none', 'source_smiles': 'NA',
                'source_inchikey': 'NA', 'parent_smiles': 'NA',
                'parent_inchikey': 'NA', 'connectivity_key': 'NA',
                'parent_is_neutral': 'NA', 'heavy_atom_count': 'NA'}
        if not rec:
            base['resolution_status'] = 'unresolved_no_chebi_structure'
            rows.append(base)
            continue
        smi, ich, ikey, _ = rec
        base['source_smiles'] = smi or 'NA'
        base['source_inchikey'] = ikey or 'NA'
        if '*' in (smi or ''):
            # ChEBI class entry with an R-group wildcard: not a discrete ligand
            base['resolution_status'] = 'unresolved_generic_class_wildcard'
            rows.append(base)
            continue
        psmi, pkey, route, ha = standardize(smi, ich)
        if not pkey:
            base['resolution_status'] = 'unresolved_rdkit_standardization_failed'
            rows.append(base)
            continue
        base.update({'resolution_status': 'resolved_exact_chebi_structure',
                     'structure_route': route, 'parent_smiles': psmi,
                     'parent_inchikey': pkey, 'connectivity_key': pkey.split('-')[0],
                     'parent_is_neutral': 'true' if pkey.endswith('-N') else 'false',
                     'heavy_atom_count': str(ha)})
        rows.append(base)
    C.write_tsv(os.path.join(C.DATA, 'CHEBI_STRUCTURE_RESOLUTION.tsv'), rows, FIELDS)

    ok = [r for r in rows if r['resolution_status'] == 'resolved_exact_chebi_structure']
    pilot = defaultdict(set)
    for r in cands:
        if r['pilot_parent_inchikey'] != 'NA':
            pilot[r['chebi_id']] |= set(r['pilot_parent_inchikey'].split(';'))
    byid = {r['chebi_id']: r for r in ok}
    agree = [c for c in pilot if c in byid and byid[c]['parent_inchikey'] in pilot[c]]
    dis = [c for c in pilot if c in byid and byid[c]['parent_inchikey'] not in pilot[c]]
    C.write_tsv(os.path.join(C.DATA, 'PILOT_STRUCTURE_CROSSCHECK.tsv'),
                [{'chebi_id': c, 'chebi_parent_inchikey': byid[c]['parent_inchikey'],
                  'pilot_parent_inchikey': ';'.join(sorted(pilot[c])),
                  'agreement': 'agree' if c in set(agree) else 'disagree'}
                 for c in sorted(set(agree) | set(dis))],
                ['chebi_id', 'chebi_parent_inchikey', 'pilot_parent_inchikey', 'agreement'])

    from collections import Counter
    print('PASS resolved=%d/%d' % (len(ok), len(rows)))
    for k, v in Counter(r['resolution_status'] for r in rows).most_common():
        print('   %-45s %5d' % (k, v))
    for k, v in Counter(r['structure_route'] for r in ok).most_common():
        print('   route %-39s %5d' % (k, v))
    print('   pilot cross-check agree=%d disagree=%d (%.1f%% agreement)'
          % (len(agree), len(dis), 100.0 * len(agree) / max(1, len(agree) + len(dis))))


if __name__ == '__main__':
    main()
