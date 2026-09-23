"""Prepare the analysis-only publication subset from the verified v1 snapshot.

Explicit manifest members only; no recursive discovery, Git operations, network
requests or scientific analysis. Source implementations are copied unchanged.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent/'secondsite_publication_code_v1'
EXCLUDED_PACKAGES = {'secondsite_web_handoff', 'secondsite_biolip_consensus60_release_v1', 'manuscript_figures'}
EXCLUDED_FILES = {'analysis/allosteric_pair_benchmark_main/scripts/render_main_figure.py'}
MEMBERS = {'scripts/build_publication_subset.py'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(name, content):
    p=ROOT/name
    p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(content,bytes): p.write_bytes(content)
    else: p.write_text(content)
    MEMBERS.add(name)

def dump(name,obj): write(name,json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def copy(name): write(name,(PRIOR/name).read_bytes())

def tsv(name,fields,rows):
    import io
    out=io.StringIO(newline='')
    w=csv.DictWriter(out,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
    write(name,out.getvalue())

def main():
    if (ROOT/'source').exists() or (ROOT/'manifests/CHECKSUMS.sha256').exists():
        raise FileExistsError('Output already populated; preserve this version.')
    original={}
    for line in (PRIOR/'manifests/CHECKSUMS.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(PRIOR/name)==digest,name
        original[name]=digest
    prior_manifest_hash=sha(PRIOR/'manifests/CHECKSUMS.sha256')
    rows=list(csv.DictReader((PRIOR/'provenance/SOURCE_INDEX.tsv').open(),delimiter='\t'))
    removed=[r for r in rows if r['stage'] in {'09','10'} or r['source_path'] in EXCLUDED_FILES]
    kept=[r for r in rows if r not in removed]
    for r in kept:
        assert r['package'] not in EXCLUDED_PACKAGES
        copy(r['release_path'])
        assert sha(ROOT/r['release_path'])==r['sha256']
    tsv('provenance/SOURCE_INDEX.tsv',list(rows[0]),kept)
    tsv('provenance/EXCLUDED_FILES.tsv',['stage','source_path','reason'],[
        dict(stage=r['stage'],source_path=r['source_path'],reason='Outside publication scope: dedicated website export or figure material') for r in removed])
    keep_source={r['source_path'] for r in kept}
    for name in ['IMPORTS.tsv','PORTABILITY_AND_NETWORK_AUDIT.tsv','DEPENDENCY_REVIEW.tsv']:
        with (PRIOR/'provenance'/name).open() as f:
            reader=csv.DictReader(f,delimiter='\t');fields=reader.fieldnames
            selected=[r for r in reader if r['file'] in keep_source]
        tsv('provenance/'+name,fields,selected)
    stages=[s for s in json.loads((PRIOR/'workflow/STAGES.json').read_text()) if int(s['stage'])<=8]
    retained_paths={r['release_path'] for r in kept}
    for s in stages:
        s['files']=[p for p in s['files'] if p in retained_paths]
        text='# '+s['stage']+'. '+s['title']+'\n\n'+s['description']+'\n\n'
        text+='Original implementation names are retained for source provenance. These files are provided for methodological inspection, not as a self-contained execution environment.\n\n'
        for path in s['files']:
            text+='- ['+str(Path(path).relative_to('source/analysis'))+'](../../'+path+')\n'
        write('workflow/'+s['stage']+'/README.md',text)
    dump('workflow/STAGES.json',stages)
    for name in ['.gitignore','tests/test_release.py']:
        copy(name)
    citation=(PRIOR/'CITATION.cff').read_text().replace('0.1.0-prepublication','0.2.0-prepublication').replace('ranking comparison and resource exports.','and ranking comparison.')
    write('CITATION.cff',citation)
    validator=(PRIOR/'scripts/validate_release.py').read_text()
    validator=validator.replace("    'secondsite_web_handoff/scripts/build_four_cohort_handoff.py',\n",'').replace('range(1, 11)','range(1, 9)')
    validator=validator.replace("    indexed = {r['release_path'] for r in rows}","""    excluded_packages = {'secondsite_web_handoff', 'secondsite_biolip_consensus60_release_v1', 'manuscript_figures'}
    assert not any(r['package'] in excluded_packages for r in rows)
    assert not any(r['source_path'].endswith('/render_main_figure.py') for r in rows)
    for package in excluded_packages:
        assert not (root/'source/analysis'/package).exists(), package
    assert not (root/'workflow/09').exists() and not (root/'workflow/10').exists()
    indexed = {r['release_path'] for r in rows}""")
    write('scripts/validate_release.py',validator)
    write('scripts/workflow.py',(PRIOR/'scripts/workflow.py').read_text().replace('range(1, 11)','range(1, 9)'))
    write('README.md', '''# SecondSite analysis code

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
''')
    workflow=(PRIOR/'docs/WORKFLOW.md').read_text().split('## 09. Export to SecondSite')[0].rstrip()+'\n'
    write('docs/WORKFLOW.md',workflow)
    data=(PRIOR/'docs/DATA_REQUIREMENTS.md').read_text()
    data='\n'.join(line for line in data.splitlines() if not line.startswith(('| SecondSite exports |','| Figures |')))+'\n'
    data=data.replace('derived inputs and numerical figure tables','derived analysis inputs').replace('code package alone does not establish raw-data-to-manuscript reproducibility','code package is intended for source inspection rather than a complete raw-data-to-manuscript rerun')
    write('docs/DATA_REQUIREMENTS.md',data)
    methods=(PRIOR/'docs/MATERIALS_AND_METHODS_CODE_MAP.md').read_text()
    methods='\n'.join(line for line in methods.splitlines() if not line.startswith('| Web dissemination |'))+'\n'
    write('docs/MATERIALS_AND_METHODS_CODE_MAP.md',methods)
    write('docs/CODE_AVAILABILITY.md','''# Code availability

Suggested statement after publication of the repository:

> Source code for reference preprocessing, dataset construction, model training and evaluation, ChEMBL screening, structural follow-up, BioLiP pair ranking and ranking comparison is available at [GitHub repository URL], with the manuscript-associated release archived at [software archive DOI]. The release includes workflow documentation and source provenance. External data and pretrained encoder resources remain subject to their respective availability and licensing terms.

Do not claim that dedicated website export/deployment or manuscript figure code is included. Until upload and archiving are complete, retain future tense and do not invent repository or archive identifiers.
''')
    write('docs/REPRODUCIBILITY.md','''# Source provenance and validation scope

Analysis implementations are byte-identical to the retained files in the verified v1 source snapshot. `provenance/SOURCE_INDEX.tsv` records original relative paths, stage assignments and hashes. `EXCLUDED_FILES.tsv` lists the intentionally omitted files; it does not contain their source code.

Fresh checks validate source hashes, Python and shell syntax, eight-stage membership, exclusion of the dedicated website-export and manuscript-figure packages, and final archive integrity. Package unit tests are rerun. Earlier scientific synthetic test results are retained separately with explicit provenance; they are not presented as newly executed scientific analyses.

This is a publication source snapshot, not a portable runtime distribution. Original workspace paths, external dependencies and historical experiment contracts remain in the source. Portability and dependency audit tables identify known references; dynamic imports are not exhaustively resolved. No analysis is rerun and no network or GitHub operation is performed when preparing this subset.

Some historical package documentation and integrated report functions mention figures or tabular exports. They are preserved with their analysis modules for provenance; the dedicated publication-figure and SecondSite export packages are intentionally absent. No website deployment configuration or manuscript figure assets are distributed.
''')
    write('docs/RELEASE_CHECKLIST.md','''# Public source-release checklist

- Verify the source and archive checksums.
- Confirm an author-approved software license and third-party code permissions.
- Supply the GitHub repository URL and tagged release identifier.
- Archive the public release and add its DOI to Code availability.
- Describe external data requirements without implying that data or trained weights are bundled.
- Keep dedicated website export/deployment and manuscript figure materials outside this public repository.

A clean-machine rerun is not a condition asserted for this code-inspection release. Publishing the snapshot does not establish end-to-end computational reproduction.
''')
    prior_tests=json.loads((PRIOR/'validation/TEST_RESULTS.json').read_text())
    dump('validation/PRIOR_SCIENTIFIC_TESTS.json',dict(executed_in_this_revision=False,
        source_release='secondsite_publication_code_v1',source_report_sha256=sha(PRIOR/'validation/TEST_RESULTS.json'),
        retained_analysis_source_unchanged=True,suites=[r for r in prior_tests if r['test']!='release_integrity_unittests']))
    spec=importlib.util.spec_from_file_location('subset_validator',ROOT/'scripts/validate_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,env=env,text=True,capture_output=True)
    dump('validation/PACKAGE_TESTS.json',dict(status='PASS' if proc.returncode==0 else 'FAIL',returncode=proc.returncode,
        tests=8,stdout=proc.stdout,stderr=proc.stderr))
    if proc.returncode: raise RuntimeError(proc.stderr)
    report=module.validate(ROOT,require_manifest=False)
    dump('validation/RELEASE_VALIDATION.json',report)
    dump('provenance/SNAPSHOT.json',dict(parent_release=PRIOR.name,parent_manifest_sha256=prior_manifest_hash,
        source_files=len(kept),code_files=sum(r['source_path'].endswith(('.py','.sh')) for r in kept),
        omitted_files=len(removed),retained_sources_byte_identical=True,upstream_modified=False))
    dump('validation/STATUS.json',dict(status='ANALYSIS_ONLY_PUBLICATION_SUBSET_VALIDATED',stages=8,
        github_upload='not_performed',git_operations_performed=False,scientific_analyses_rerun=False,
        dedicated_secondsite_export_included=False,manuscript_figures_included=False,
        original_package_preserved=True))
    names=sorted(MEMBERS)
    write('manifests/CHECKSUMS.sha256',''.join(sha(ROOT/p)+'  '+p+'\n' for p in names))
    frozen={p:sha(ROOT/p) for p in MEMBERS}
    final=module.validate(ROOT,require_manifest=True)
    subprocess.run(['sha256sum','--check','--quiet','manifests/CHECKSUMS.sha256'],cwd=ROOT,check=True)
    assert frozen=={p:sha(ROOT/p) for p in MEMBERS}
    assert original=={p:sha(PRIOR/p) for p in original}
    assert prior_manifest_hash==sha(PRIOR/'manifests/CHECKSUMS.sha256')
    archive=ROOT.parent/(ROOT.name+'.zip')
    if archive.exists(): raise FileExistsError(archive)
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for name in sorted(MEMBERS): z.write(ROOT/name,ROOT.name+'/'+name)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(MEMBERS)
        for name in MEMBERS:
            assert hashlib.sha256(z.read(ROOT.name+'/'+name)).hexdigest()==frozen[name]
        assert not any('/source/analysis/'+p+'/' in n for p in EXCLUDED_PACKAGES for n in z.namelist())
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(final,omitted_files=len(removed),archive=str(archive),
        archive_bytes=archive.stat().st_size,prior_package_unchanged=True),indent=2))

if __name__=='__main__': main()
