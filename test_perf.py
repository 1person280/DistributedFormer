import sys, time, numpy as np
sys.path.insert(0, '.')
from distributedformer.core.distributedformer import DistributedFormer

df = DistributedFormer(depth=1, dim=16, num_think_layers=3, training_mode=True)
df.enable_learning(False)
for _ in range(2):
    df.step(np.random.randn(16)*0.5)

s = time.time()
for _ in range(10):
    df.step(np.random.randn(16)*0.5)
print("10 steps (no STDP): %.3fs" % (time.time()-s))

df2 = DistributedFormer(depth=1, dim=16, num_think_layers=3, training_mode=True)
df2.enable_learning(True)
for _ in range(2):
    df2.step(np.random.randn(16)*0.5)

s = time.time()
for _ in range(10):
    df2.step(np.random.randn(16)*0.5)
print("10 steps (with STDP): %.3fs" % (time.time()-s))
