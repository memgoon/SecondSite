# Methods-to-code map

| Methods component | Implementation | Reproduction detail to retain |
|---|---|---|
| Reference integration | Preprocessing and participant/KEGG source packages | Source snapshot, exact protein/compound identity, conflict handling and manual decisions |
| Model inputs | Main/broad input preparation and alignment utilities | Selected chain, atom and residue indices, encoder versions and tensor checksums |
| Dataset construction | Base matrix contract and ligand-anchored builder | Every-pair, protein-, ligand- and double-anchored membership; no extra balancing implied by anchoring |
| Model fitting | Base trainer and explicit/ligand/LOFO extensions | Same eight architectures, seed/fold identities, validation selection versus fixed-epoch protocol |
| Held-out evaluation | Split and aggregate utilities | Connectivity versus Murcko scaffold, protein-family exclusion and purged double held-out partitions |
| Statistical analysis | Aggregate/bootstrap programs | Metric, eligible rows/groups, paired comparison and resampling unit |
| ChEMBL deployment | Deployment trainers and four-cohort completion | Epoch provenance, checkpoint hashes, identical-row comparisons and model-specific input availability |
| Structural follow-up | Competitive docking package | Receptor preparation, ligand charge, box definitions, scoring function and random seed |
| BioLiP ranking | Exact reference linkage and consensus-60 pipeline | One exact protein–ligand pair, per-PDB weighted consensus, geometry diagnostics, held-out refitting and score interpretation |
| Cross-method comparison | BioLiP neural ranking comparison | Exact pair linkage, common evaluation support and paired cluster uncertainty |

This map describes implementation locations. The scientific narrative and numbers remain governed by the manuscript and its frozen numerical outputs, not by old package READMEs. The packaging process does not add analyses or change thresholds.
