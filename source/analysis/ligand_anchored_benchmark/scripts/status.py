"""Current run progress from 480 known report paths, not historical logs."""
import json
import time
from collections import Counter
from common import PACKAGE,jobs,compatible_report
from queue_jobs import connect


def main():
    path=PACKAGE/'gpu_output/RUN_CONTRACT.json'
    if not path.exists():
        print('No run started. Expected 480 fits.')
        return
    run=json.loads(path.read_text())
    counts=Counter()
    total=0
    for job in jobs():
        if compatible_report(job,run,hashes=False):
            counts[job[0]]+=1
            total+=1
    print('Completed current-run fits: %d / 480 (%.1f%%)'%(total,100.*total/480))
    print(dict(counts))
    if (PACKAGE/'gpu_output/queue.sqlite').exists():
        with connect() as c:
            for job,worker in c.execute("SELECT job,worker FROM jobs WHERE status='running'"):
                p=PACKAGE/('gpu_output/progress/worker%s.json'%worker)
                progress=json.loads(p.read_text()) if p.exists() else {}
                print('worker=%s job=%s epoch=%s last_update_age_seconds=%.0f'%(worker,job,
                    progress.get('epoch','starting'),time.time()-progress.get('updated',time.time())))
            print('Queue states:',dict(c.execute('SELECT status,count(*) FROM jobs GROUP BY status')))
    print('Counts are lightweight report checks; final validation checks artifact SHA256.')


if __name__=='__main__':
    main()

