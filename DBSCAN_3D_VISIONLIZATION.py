import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

da = []
num = 0
for line in open('/home/jungmin/바탕화면/OpenKE/benchmarks/LUMB/Triple2Vector/rescal_Triple2Vector_lubm.txt'):
    num = num + 1
    if num < 3000:
        da.append([float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2])])

data = np.array(da)

model = DBSCAN(eps=0.05, min_samples=1)
model.fit_predict(data)
pred = model.fit_predict(data)

print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)
centroid = []
for i in range(len(set(model.labels_))):
    centroid_0 = []
    centroid_1 = []
    centroid_2 = []
    print(i)
    for idx, val in enumerate(model.labels_ == i):
        if val:
            centroid_0.append(data[idx][0])
            centroid_1.append(data[idx][1])
            centroid_2.append(data[idx][2])
    centroid.append([np.mean(centroid_0), np.mean(centroid_1), np.mean(centroid_2)])
    centroid_0.clear()
    centroid_1.clear()
    centroid_2.clear()

print(centroid)

cent = np.array(centroid)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=model.labels_, s=5)
ax.scatter(cent[:, 0], cent[:, 1], cent[:, 2], s=50, marker='*', c='red')
ax.view_init(azim=200)
ax.set_xlabel('subject')
ax.set_ylabel('predicate')
ax.set_zlabel('object')
plt.show()
