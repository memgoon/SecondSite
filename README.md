# SecondSite analysis code

Source code accompanying **SecondSite: a benchmark and web resource for prioritizing candidate allosteric protein–ligand interactions**.

This publication snapshot contains the scientific analysis code, arranged in workflow order. It is intended for methodological inspection and code availability, not as a one-command execution package. Original implementation names and source hashes are retained.

## Included workflow

| Step | Analysis |
|---|---|
| [01](workflow/01/README.md) | Preprocessing and reference integration |
| [02](workflow/02/README.md) | Structural inputs and dataset construction |
| [03](workflow/03/README.md) | Benchmark training and evaluation |
| [04](workflow/04/README.md) | Sensitivity analyses and uncertainty |
| [05](workflow/05/README.md) | ChEMBL screening and evaluation |
| [06](workflow/06/README.md) | Exploratory structural follow-up |
| [07](workflow/07/README.md) | BioLiP reference linkage and pair ranking |
| [08](workflow/08/README.md) | Neural and structural ranking comparison |

Dedicated SecondSite export/deployment packages and manuscript figure code, assets and instructions are **not included**. Analysis modules are otherwise unchanged; some contain routine diagnostic plotting or tabular-output functions integral to the analysis. These do not constitute the separately excluded figure or website-release packages.

## Verify the snapshot

```bash
python scripts/validate_release.py
python -m unittest discover -s tests -v
sha256sum --check --quiet manifests/CHECKSUMS.sha256
```

These checks do not run a scientific analysis or access an external service. See [WORKFLOW.md](docs/WORKFLOW.md), the [Methods–code map](docs/MATERIALS_AND_METHODS_CODE_MAP.md), [data requirements](docs/DATA_REQUIREMENTS.md) and [provenance notes](docs/REPRODUCIBILITY.md).

The source retains original data paths and experiment contracts. External datasets, encoder weights, trained models and server credentials are not distributed. No clean-machine end-to-end reproduction is claimed. Repository URL, license and archived release DOI remain to be supplied before public release; no GitHub upload has been performed.
