import numpy as np
from cluster import Clusters

data_3d = np.array([[0,0,0],
                        [1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [1,1,0],
                        [1,0,1],
                        [0,1,1],
                        [1,1,1],
                        [1,1,2],
                        [1,1,1.5],
                        [0.5,0,0],
                        [0,0.5,0],
                        [0,0,0.5]])
K = 3
cl = Clusters(data_3d, K)
cl.clustering(visual=False)
print(f"There are {K} clusters.\nThe centroids are\n", cl.centroids)