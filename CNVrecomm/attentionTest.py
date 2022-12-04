import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


a=np.random.randint(-1,2,(3,3))
print(a)

a=np.matmul(a,a.T)
print(a)

import torch.nn.functional as tnf
a=torch.tensor(a,dtype=torch.float32)
b=tnf.softmax(a,dim=1)
print(b)

b=b.numpy()
a=a.numpy()
c=np.matmul(b,a)
print(c)

#变种注意力
print("变种注意力===============")
a=[1,-1,2,0,-1,-1]#定义的常数。
v=np.random.randint(-1,2,(3,3))
print(v)

v=torch.tensor(v,dtype=torch.float32)
print(v)
v_r=v.repeat_interleave(3,0)
print(v_r)
v_rr=v.repeat(3,1)
print(v_rr)
vv=torch.cat((v_r,v_rr),dim=1)
print(vv)
