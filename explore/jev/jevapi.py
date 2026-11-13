"""Minimal client for TypeSafe's System One API (Jev) with an on-disk cache and thread-pool concurrency.
Key from $TYPESAFE_API_KEY or /home/user/.secrets/jev_key (never stored in the repo)."""
import os, json, hashlib, time, threading, requests
from concurrent.futures import ThreadPoolExecutor
URL = "https://api.typesafe.ai/v1/systemone"
KEY = os.environ.get("TYPESAFE_API_KEY") or open("/home/user/.secrets/jev_key").read().strip()
MODEL = os.environ.get("JEV_MODEL", "jev-1.14.0")
_lock = threading.Lock()
_sess = threading.local()


def _session():
    if not hasattr(_sess, "s"):
        _sess.s = requests.Session()
    return _sess.s


def _key(req):
    return hashlib.sha1(json.dumps(req, sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line); self.d[r["k"]] = r["v"]

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with _lock:
            self.d[k] = v
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")


def call(state, questions, cache=None, model=MODEL, tries=6):
    req = {"model": model, "state": state, "questions": questions}
    k = _key(req)
    if cache is not None and (v := cache.get(k)) is not None:
        return v
    for t in range(tries):
        try:
            r = _session().post(URL, json=req, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
            if r.status_code == 200:
                v = r.json()
                if cache is not None:
                    cache.put(k, v)
                return v
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 ** t); continue
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            time.sleep(2 ** t)
    raise RuntimeError("too many retries")


def call_many(jobs, cache=None, workers=16, model=MODEL, log_every=500):
    """jobs: list of (state, questions). Returns list of responses (or exceptions)."""
    out = [None] * len(jobs); t0 = time.time(); done = [0]

    def run(i):
        try:
            out[i] = call(*jobs[i], cache=cache, model=model)
        except Exception as e:
            out[i] = e
        with _lock:
            done[0] += 1
            if done[0] % log_every == 0:
                print(f"{done[0]}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, range(len(jobs))))
    return out
