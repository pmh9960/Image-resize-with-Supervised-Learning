import h5py
import numpy as np

f = h5py.File("data/test.hdf5", "w")

# group
g = f.create_group("group")
sub = g.create_group("sub")

print(f.keys())
print(g.keys())


# attributes
sub.attrs["desc"] = "hello, world"
print(sub.attrs["desc"])

# 아래 둘 다 가능
a = sub.create_dataset("a", data=np.arange(20))
b = sub["b"] = np.arange(10)

print(sub.keys())

"""
f
|
g (name=group, one of f.keys())
|
sub (name=sub, one of g.keys())
    (attrs["desc"]="hello, world")
|
(name=a, data_a), (name=b, data_b)
"""

for k in sub.keys():
    dataset = sub[k]
    print(dataset)
    print(dataset.dtype)
    print(dataset[()])

# Access dataset
b = f[u"/group/sub/b"]
print(b)

