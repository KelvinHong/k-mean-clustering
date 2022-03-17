import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette as palette # For color pallettes 
from scipy.spatial import ConvexHull

class Clusters():
    def __init__(self, data, K):
        assert len(data.shape) == 2, "data should be a 2-dimensional numpy array."
        self.data = data
        assert isinstance(K, int), "Number of cluster should be a positive integer. " 
        self.K = K
        self.dim = data.shape[1]
        self.palette = palette(None, self.K)
    
    def __len__(self):
        return self.data.shape[0]

    def visualize(self, title=None):
        if self.dim != 2:
            print("Warning: Visualization is only enabled with 2-dimensional data.")
            return  
        else:
            xy = self.data.transpose()
            if not (hasattr(self, "centroids") and hasattr(self, "classes")):
                # Show points
                plt.title("Scatter plot without centroids")
                plt.scatter(xy[0], xy[1])
                plt.show()
            else:
                # Show points with cluster links
                if title:
                    plt.title(title)
                else:
                    plt.title("Scatter plot with centroids")
                plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=30, c = self.palette, marker="^")
                colors = [self.palette[int(self.classes[i])] for i in range(len(self))]
                plt.scatter(self.data[:, 0], self.data[:, 1], c = colors, marker="o")
                # Plot convex hull for each group
                for i, palette in enumerate(self.palette):
                    points = self.data[self.classes == i]
                    if points.shape[0] > 2:
                        hull = ConvexHull(points)
                        for simplex in hull.simplices:
                            plt.plot(points[simplex, 0], points[simplex, 1], color=palette)
                plt.show()

    
    def clustering(self, visual=False, force_stop = 100):
        """
        The algorithm will be terminated in force_stop iterations.
        This will store the clusters' centroids in self.centroids.
        """
        # Initialization
        self.centroids = self.data[np.random.choice(len(self), size=self.K, replace=False)]
        n = len(self)
        self.classes = np.zeros((n)) - 1
        if visual: self.visualize(title="Original points")
        i = 0
        # Flag for whether classes change after one iteration
        changing = True 
        # Start iterating
        while i <= force_stop and changing:
            # Copy classes for change detection
            new_classes = self.classes.copy()
            # Calculate difference from n points to K centroids
            diffs = self.data[..., np.newaxis] - self.centroids.transpose()[np.newaxis, ...] # Shape (n, 2, K)
            diffs = np.sqrt(diffs[:, 0, :] ** 2 + diffs[:, 1, :] ** 2) # Shape (n, K)
            # New classes assignment
            new_classes = np.argmin(diffs, axis=1) # Shape (n)
            changing = np.sum(np.abs(new_classes - self.classes)) != 0
            if not changing:
                break
            self.classes = new_classes.copy()
            # Re-calculate centroid based on clusters
            for j in range(self.K):
                bool_select = self.classes == j
                self.centroids[j] = np.mean(self.data[bool_select], axis=0)
            if visual: self.visualize()
            i += 1
        print(f"Clustering completed. Used {i} iterations.")
        print("The groups are", self.classes)


if __name__ == "__main__":
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
    K = 2
    cl = Clusters(data_2d, K)
    cl.clustering(visual=True)
    cl.visualize(title = "Final result")
    print(f"There are {K} clusters.\nThe centroids are\n", cl.centroids)