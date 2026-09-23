"""Small transactional queue; each process claims a distinct complete fit."""
import json
import sqlite3
import time
from common import PACKAGE, jobs, compatible_report, partition, read_data


def connect():
    c = sqlite3.connect(str(PACKAGE / 'gpu_output/queue.sqlite'), timeout=120)
    c.execute('PRAGMA busy_timeout=120000')
    return c


def initialize(run):
    frame = read_data()
    sizes = {(r, f): len(partition(frame, r, f)[0]) for r in ('family_only', 'double_unseen') for f in range(47)}
    cost = dict(ligand=1, protein=2, c1=3, c2=6, c3=15, d1=2, d2=3, d3=5)
    with connect() as c:
        c.execute('CREATE TABLE IF NOT EXISTS jobs (job TEXT PRIMARY KEY, priority REAL, status TEXT, worker TEXT, updated REAL)')
        c.execute('DELETE FROM jobs')
        for job in jobs():
            done = compatible_report(job, run) is not None
            c.execute('INSERT INTO jobs VALUES (?,?,?,?,?)', (json.dumps(job), sizes[(job[0], job[3])]*cost[job[1]], 'done' if done else 'pending', '', time.time()))


def claim(worker):
    with connect() as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute("SELECT job FROM jobs WHERE status='pending' ORDER BY priority DESC, job LIMIT 1").fetchone()
        if row is None:
            return None
        c.execute("UPDATE jobs SET status='running',worker=?,updated=? WHERE job=?", (worker, time.time(), row[0]))
        return tuple(json.loads(row[0]))


def finish(job, status):
    with connect() as c:
        c.execute('UPDATE jobs SET status=?,updated=? WHERE job=?', (status, time.time(), json.dumps(job)))
