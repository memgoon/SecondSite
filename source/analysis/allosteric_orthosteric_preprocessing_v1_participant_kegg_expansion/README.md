# Participant/KEGG orthosteric expansion

This append-only preprocessing package adds orthosteric reference pairs carrying
curated accession-level reaction-participant identity **without** requiring a
co-crystal of that accession with that compound. It modifies no prior package.

Principal outputs:

- `data/EXPANSION_STRICT_ORTHOSTERIC_PAIRS.tsv` - 1,677 registered pairs across
  336 proteins, spanning 674 distinct chemicals;
- `data/EXPANSION_PAIR_LEDGER.tsv` - one final decision for each of the 2,830
  in-scope protein-ChEBI candidates;
- `data/UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_PARTICIPANT_KEGG_AUGMENTED.tsv.gz` -
  20,668 rows, the 18,991 Pilot 4 rows preserved verbatim plus the new pairs;
- `data/EXPANSION_DEPTH_METRICS.tsv` and `ALLOSTERIC_SIDE_DEPTH_CEILING.tsv`;
- `data/AUDIT_EXCLUDED_BIOLIP_ROWS.tsv` - the 19 rows the strict-pair multisite
  audit removed, listed rather than deleted from any prior package;
- `validation/VALIDATION.json` - independent 16-gate validation.

Protein anchoring rises from 313 to **449** of 635 allosteric proteins. Median
total rows per anchored protein rises from 3 to 5, and proteins with matched
within-protein depth >=3 rise from 19 to 71, against a ceiling of 148 set by the
allosteric side.

## Scope is applied by chemical identity, not by lane

Filtering the KEGG lane by compound identifier and the participant lane by
UniProt evidence type is not sufficient: UniProt annotates water, protons and
metal ions as `catalytic_reaction_participant`, so an early build registered the
proton, water and eight bare metal ions as orthosteric ligands. Exclusions are
now applied identically to both lanes at the chemical level, a discrete ligand
must carry more than one heavy atom, and both exclusion sets are closed over
ChEBI conjugate acid/base and tautomer relations so charged annotation forms
such as NAD(1-) and coenzyme A(4-) are caught.

## Why no co-crystal is required

The frozen BioLiP pilots registered a pair only when the compound was observed
bound in a structure of that exact accession, discarding 1,367 UniProt
participant seeds. Those 281 proteins all have BioLiP structures (median 5
deposited ligand CCDs, 30 exact observations each); what is missing is a
structure of that protein with that specific compound. Since the label unit is
the protein-ligand pair rather than the site, the gate was removing evidence
rather than enforcing it, and it biased the orthosteric class toward
crystallisable chemistry.

## Stricter than the pilots in one respect

Registration accepts standardized-parent chemical identity, so the ASD conflict
test is applied at both the full InChIKey and the connectivity-key level. The
frozen pilots tested only the full InChIKey. Fifteen candidates were rejected by
the connectivity test alone.

## Running

See `CODE_AVAILABILITY.md`. Reproduction needs no network access; the ChEBI,
KEGG and UniProt payloads are cached under `cache/` and hashed in the manifest.

```bash
sha256sum -c manifests/CHECKSUMS.sha256
```

## Interpretation limit

`orthosteric` here denotes curated accession-specific participation in a
canonical reaction. It does not assign a binding site, potency,
competitive/noncompetitive mechanism, or an allostery conclusion. These rows are
annotation-grounded without a co-crystal, so the same-chemical multisite audit
applied to the structural pairs cannot be run on them; that is a known
specificity gap of this tier. The augmented table is an uncontrolled pair
universe and has not been split or merged into a training design.
