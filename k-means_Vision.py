import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

array = []
centNum = 10
for line in open("/home/jungmin/바탕화면/OpenKE/benchmarks/LUMB/Triple2Vector/distmult_Triple2Vector_lubm.txt", encoding='utf-8', mode='r'):
    sub, pred, obj = line.split(',')
    array.append([sub, pred, obj])

array = np.array(array).astype(float)


clf = KMeans(n_clusters = 10)
pre = clf.fit_predict(array)



#visionlization
colors = ['#1f77b4', '#ff7f0e', '#8c564b','#06623B','#2ca02c', '#d62728', '#9467bd',  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
fig = plt.figure()
ax = Axes3D(fig)


for i in range(centNum):
    ax.scatter(array[pre == i, 0], array[pre == i, 1], array[pre == i, 2], s=5, c=colors[i], marker='o',  label='cluster ' + str(i))

# plot the centroids
ax.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], clf.cluster_centers_[:, 2], s=50, marker='*', c='red', label='centroid')

ax.legend(scatterpoints=1)

# ax.title('K-Means')

ax.set_title('TransD')

ax.set_xlabel('subject')
ax.set_ylabel('predicate')
ax.set_zlabel('object')
# plt.grid()
plt.show()


