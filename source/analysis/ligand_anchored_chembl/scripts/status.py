"""Count completed artifacts directly, never infer completion from log line counts."""
from shared import *
completed=[];inferred=[]
for m in MODELS:
    for seed in SEEDS:
        p=deploy_dir(m,seed)/'DEPLOY_REPORT.json'
        if p.exists() and read_json(p).get('status')=='validated':completed.append((m,seed))
for scope,i in tasks():
    p=output(scope,i).with_suffix('.json')
    if p.exists() and read_json(p).get('status')=='validated':inferred.append((scope,i))
print('Deployment reports: %d/24; completed inference tasks: %d/9'%(len(completed),len(inferred)))
for m in MODELS:print('  %s: %d/3 seeds'%(m,sum(x[0]==m for x in completed)))
print('Inference:',inferred)
print('Report counts only; hashes are validated by resume/return/CPU aggregation.')
