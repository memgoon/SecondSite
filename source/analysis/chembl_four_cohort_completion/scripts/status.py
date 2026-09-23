"""Read only expected completion reports, not tensors or recursive directories."""
from collections import Counter
from common import *

def main():
    ts=tasks();path=PACKAGE/'gpu_output/RUN_CONTRACT.json'
    if not path.exists():print('Preflight not completed: 0/%d inference chunks'%len(ts));return
    run=read_json(path);done=Counter();expected=Counter()
    for t in ts:
        key=t['arm']+'/'+t['model'];expected[key]+=1
        if complete(t,run,check_hash=False):done[key]+=1
    print('Completed chunks: %d/%d (not equal-cost units)'%(sum(done.values()),len(ts)))
    for key in sorted(expected):print('%-24s %d/%d'%(key,done[key],expected[key]))

if __name__=='__main__':main()
