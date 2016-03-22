import random
import itertools as it
from sklearn.externals import joblib

def simulate_loads(nfrag, ngt, q):
    loads = [1] * nfrag
    active = set(range(nfrag))
    for k in range(1, len(loads)):
        i0, i1 = random.sample(active, k=2)
        active.remove(i0)
        active.remove(i1)
        active.add(len(loads))
        if random.random() > q:  # correct merge
            new_load = max(loads[i0], loads[i1])
        else:  # false merge
            new_load = min(loads[i0] + loads[i1], ngt)
        loads.append(new_load)
    return loads


def many_sims(n_jobs=2):
    qs = [.025, .05, .1, .2]
    nfrags = [10000, 20000, 40000, 80000, 160000]
    nreps = 5
    keys = [(n, q) for n, q, i in it.product(nfrags, qs, range(nreps))]
    results = joblib.Parallel(n_jobs=n_jobs)(
                  joblib.delayed(simulate_loads)(n, 1000, q) for n, q in keys
              )
    return dict(zip(keys, results))
