# External data and artifacts

| Stage | Inputs required | Redistribution in this package |
|---|---|---|
| Reference integration | Frozen ASD labels; GtoPdb and KLIFS annotations; cached BRENDA K03 tables and resolved compounds; UniProt, ChEBI and KEGG participant mappings; manual conflict/exclusion decisions | Not included |
| Molecular inputs | Exact pair membership, canonical ligand structures, selected PDB receptor chains, UniProt mappings, Pfam components, pocket residue indices, frozen protein and atom embeddings | Not included |
| Benchmark | Model-ready tables, split assignments, input tensors, experiment contracts; train/validation/test identities | Not included |
| ChEMBL | Frozen reference and screen rows, training-pair blacklists, text annotations, selected-chain/pocket availability, seed-specific deployment checkpoints and epoch provenance | Not included |
| Docking | Selected structures, receptor/ligand preparation inputs, box definitions, scoring-function binaries and seeds | Not included |
| BioLiP | Exact reference observations, AlloBench-linked reference labels, known orthosteric sites, mapped structural observations, cached coordinates, ligand properties, frozen folds, known-role checks | Not included |
| Ranking comparison | Neural and BioLiP out-of-fold scores, matched reference membership, source validation files | Not included |

Expected dataset dimensions in the reference manuscript are 6,854 every-pair, 4,637 protein-anchored, 1,365 ligand-anchored and 395 double-anchored pairs. These are provenance expectations, not substitute input data or results of a new build.

Complete reproduction requires a separately deposited, rights-cleared set of derived analysis inputs. The appropriate repository identifier is not yet supplied. Until it is, the code package is intended for source inspection rather than a complete raw-data-to-manuscript rerun. Do not redistribute private caches or entire database downloads merely to satisfy missing path checks.
