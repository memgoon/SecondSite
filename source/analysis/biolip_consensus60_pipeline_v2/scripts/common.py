import csv,gzip,hashlib,io,json,math,os,sys,time,uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED

def worker_init(parent_pid):
    # Linux: a hard-killed coordinator must not leave orphan workers holding its run lock.
    import ctypes,signal
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=parent_pid:os.kill(os.getpid(),signal.SIGTERM)
    from threadpoolctl import threadpool_limits
    global _thread_limit
    _thread_limit=threadpool_limits(limits=1)

PKG=Path(__file__).resolve().parents[1]
REPO=PKG.parents[1]
OLD=REPO/'analysis/biolip_bayesian_ranking_revision'

def versions():
    from importlib.metadata import version
    return {n:version(n) for n in ['numpy','pandas','scipy','scikit-learn','gemmi','biopython','matplotlib','threadpoolctl']}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def content(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def identity(x):return hashlib.sha256(content(x)).hexdigest()

def atomic(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.pending.'+uuid.uuid4().hex)
    with open(q,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(q,p)
    fd=os.open(str(p.parent),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def save(p,x):atomic(p,content(x)+b'\n')
def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f,delimiter='\t'))
def table(p,rs,fields=None):
    fields=fields or sorted(rs[0]);b=io.StringIO();w=csv.DictWriter(b,fields,delimiter='\t',lineterminator='\n');w.writeheader()
    for r in rs:w.writerow({k:('NA' if r.get(k) is None else r.get(k,'NA')) for k in fields})
    data=b.getvalue().encode();atomic(p,gzip.compress(data,mtime=0) if str(p).endswith('.gz') else data)
def val(x):
    if x is None or str(x).lower() in {'','na','nan','none'}:return None
    f=float(x)
    if not math.isfinite(f):raise ValueError('nonfinite input')
    return f

def tasks(out,stage,fn,items,workers,fail_after=0):
    """Bounded scheduling, deterministic ordering, one writer; durable completed tasks."""
    root=out/'tasks'/stage;root.mkdir(parents=True,exist_ok=True)
    values=[None]*len(items);pending=[];completed=0
    for i,item in enumerate(items):
        key='%05d_%s'%(i,identity(item)[:16]);p=root/(key+'.json.gz');mark=root/(key+'.commit.json')
        if mark.exists():
            m=json.loads(mark.read_text())
            if m['input']!=identity(item) or digest(p)!=m['output']:raise ValueError('committed task corruption: '+key)
            values[i]=json.loads(gzip.decompress(p.read_bytes()));completed+=1
        else:
            if p.exists():os.replace(p,p.with_name(p.name+'.orphan.'+uuid.uuid4().hex))
            pending.append((i,item,p,mark))
    save(out/'STATUS.json',dict(stage=stage,status='RUNNING',completed=completed,total=len(items)))
    print('%s: %d/%d restored'%(stage,completed,len(items)),flush=True)
    iterator=iter(pending);active={}
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(os.getpid(),)) as pool:
        def submit():
            entry=next(iterator,None)
            if entry:active[pool.submit(fn,entry[1])]=entry
        for _ in range(min(len(pending),workers*2)):submit()
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                i,item,p,mark=active.pop(future);value=future.result()
                atomic(p,gzip.compress(content(value),mtime=0));save(mark,dict(input=identity(item),output=digest(p)))
                values[i]=value;completed+=1
                save(out/'STATUS.json',dict(stage=stage,status='RUNNING',completed=completed,total=len(items)))
                if completed%10==0 or completed==len(items):print('%s: %d/%d'%(stage,completed,len(items)),flush=True)
                if fail_after and completed==fail_after:os._exit(86)
                submit()
    commits={p.name:digest(p) for p in [root/('%05d_%s.commit.json'%(i,identity(item)[:16])) for i,item in enumerate(items)]}
    save(out/'stages'/(stage+'.json'),dict(status='PASS_TASKS',tasks=len(items),commit_hashes=commits))
    return values
