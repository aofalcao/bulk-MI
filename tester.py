import numpy as np
from scipy.sparse import csr_matrix
import bulkMI

try:
    from numba import jit
except ImportError:
    print("Numba is not installed - Code is run withount JIT")


D=np.array([[0,1,0],
            [1,0,1],
            [1,1,0],
            [1,0,0],
            [0,0,0],
            [0,1,1],
            [1,0,0],
            [0,0,0]])
D=D.astype(np.float32)


#res1  = buklkMIbulk_MI_torch(torch.tensor(D, dtype=int))
res  = bulkMI.pairwise_MI(D)
np.set_printoptions(suppress=True, precision=3)
print("Pairwise")
print(res)
#r_s, r_ss, r_b, r_bo, r_bs    

res2  = bulkMI.bulk_MI_base(D)
print("Bulk Simple")
print(res2)

res3 =  bulkMI.bulk_MI(D)
print("Bulk Optimized")
print(res3)

Ds = csr_matrix(D)
res4 =  bulkMI.bulk_MI(Ds)
print("Bulk Optimized Sparse")
print(res4)


try:
    import torch
    D  = bulkMI.bulk_MI(torch.tensor(D, dtype=int))
    print("Bulk Torch")
    print(res4)
    
except ImportError:
    print("PyTorch is not installed. Test not run")
