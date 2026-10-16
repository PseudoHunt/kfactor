import sys; sys.path.insert(0, "/home/kfactor")
import kernel_fit as kf
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
kf.LM_ITERS = int(sys.argv[1])
tasks = [("llama2-7b", 1), ("llama2-7b", 15), ("llama2-7b", 24), ("qwen1.5-7b", 1), ("qwen1.5-7b", 14), ("qwen1.5-7b", 28)]
def run(t):
    kf.LM_ITERS = int(sys.argv[1]); return kf.run_layer(t)[0]
with ProcessPoolExecutor(6) as ex:
    rows = [r for res in ex.map(run, tasks) for r in res]
d = pd.DataFrame(rows); d = d[d.method == "fitted_refined"]
d.to_csv(f"/tmp/claude-0/-home/e2f83f18-0b86-4291-a141-74f6ee63f4af/scratchpad/conv_{sys.argv[1]}.csv", index=False)
print(d.to_string(index=False))
