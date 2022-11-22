import numpy as np

points = [[1,2,3],[4,5,6],[7,8,-9]]

# print(np.max(np.abs(points)))

depth_bounds = [2,5]
depth_steps = 10

depth_skipped = depth_bounds[0] / (depth_bounds[1] - depth_bounds[0]) * depth_steps
print(depth_skipped)

fis_len = 3
thetas_len=2

fi_idxes = np.zeros([fis_len, thetas_len], dtype=np.int64)
for i in range(thetas_len): #for i in range fis_len 
    fi_idxes[:, i] = i
fi_idxes = fi_idxes.flatten()

coords = [[1,1,1],[2,2,2]]
print(np.sum(coords, axis=0))

x = np.array([[1,1,1],[1,0,1]])
v = np.array([2,3])

for i in range(3):
    x[:,i]= x[:,i]* v
print(x)