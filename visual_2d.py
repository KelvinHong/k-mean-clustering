import numpy as np
from cluster import Clusters

data_2d = np.array([[0,0],
                    [1,1],
                    [0.5,1],
                    [1,0],
                    [2,3],
                    [3,2],
                    [3,3],
                    [3,3.5],
                    [0,10],
                    [1,10],
                    [0.5,9],
                    [0.5,6]])
K = 3
cl = Clusters(data_2d, K)
cl.clustering(visual=True)
cl.visualize(title = "Final result")
print(f"There are {K} clusters.\nThe centroids are\n", cl.centroids)