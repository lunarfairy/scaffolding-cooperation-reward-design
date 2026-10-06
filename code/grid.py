"""Run a list of configs with a worker pool spread over GPUs, retrying failed runs (e.g. CUDA OOM on a shared host).
usage: python3 grid.py configs.json W gpus [deltas.json] [outdir]"""
import json, os, subprocess, sys, time
cfgs = json.load(open(sys.argv[1])); W = int(sys.argv[2]); gpus = sys.argv[3].split(',')
deltas = sys.argv[4] if len(sys.argv) > 4 else ''
out = sys.argv[5] if len(sys.argv) > 5 else 'out'
os.makedirs(out, exist_ok=True); os.makedirs('logs', exist_ok=True)
todo = [c for c in cfgs if not os.path.exists(f"{out}/{c['name']}.json")]
tries = {c['name']: 0 for c in todo}
running = []
k = 0
t0 = time.time()
while todo or running:
    while todo and len(running) < W:
        c = todo.pop(0); g = gpus[k % len(gpus)]; k += 1
        tries[c['name']] += 1
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, RDP_THREADS='2', OMP_NUM_THREADS='2')
        f = open(f"logs/{c['name']}.log", 'w')
        p = subprocess.Popen([sys.executable, 'run_one.py', json.dumps(c), out, deltas], stdout=f,
                             stderr=subprocess.STDOUT, env=env)
        running.append((p, c, f))
    time.sleep(5)
    still = []
    for p, c, f in running:
        if p.poll() is None:
            still.append((p, c, f))
        else:
            f.close()
            ok = p.returncode == 0 and os.path.exists(f"{out}/{c['name']}.json")
            if not ok and tries[c['name']] < 3:
                todo.append(c)
            print(f"{time.time()-t0:7.0f}s done {c['name']} rc={p.returncode} try={tries[c['name']]} left={len(todo)}", flush=True)
    running = still
print('ALL DONE', time.time() - t0, flush=True)
