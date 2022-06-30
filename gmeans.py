import numpy as np
import scipy
from sklearn.cluster import KMeans


class GMeans:
    """
    Implementation of gmeans clustering with freely configurable significance Levels in the Anderson-Darling Test
    
    Args:
    min_obs (int): Minimum number of samples per cluster
    max_depth (int): Maximum depth of recurrence k-means
    random_state (int): Random state for k-means
    p_criteria(float): Significance Levels in the Anderson-Darling Test  
    """
    
    def __init__(self, min_obs=7, max_depth=10, random_state=None, p_criteria=0.0001):
        self.max_depth = max_depth        
        self.min_obs = min_obs
        self.random_state = random_state
        self.p_criteria = p_criteria
        
    def anderson_darling_norm_test(self, x):
        y = np.sort(x)
        xbar = np.mean(x, axis=0)
        N = len(x)
        s = np.std(x, ddof=1, axis=0)
        w = (y - xbar) / s
        logcdf = scipy.stats.norm.logcdf(w)
        logsf = scipy.stats.norm.logsf(w)
        i = np.arange(1, N + 1)
        A2 = -N - np.sum((2*i - 1.0) / N * (logcdf + logsf[::-1]), axis=0) 
        A2 = A2 * (1 + (0.75/N) + (2.25/N**2))
        
        if A2 <= 0.2:
            p_value = 1 - np.exp(-13.436 + 101.14*A2 - 223.73*(A2**2))
        elif A2 <= 0.34:
            p_value = 1 - np.exp(-8.318 + 42.796*A2 - 59.938*(A2**2))
        elif A2 < 0.6:
            p_value = np.exp(0.9177 - 4.279*A2 - 1.38*(A2**2))
        elif A2 < 10:
            p_value = np.exp(1.2937 - 5.709*A2 + 0.0186*(A2**2))
        else:
            p_value = 3.7e-24
        return p_value >= self.p_criteria
    
    def _recursiveprocess(self, data, depth, index):
        depth += 1
        if depth == self.max_depth:
            self.data_index[index[:, 0]] = index
            return
            
        km = KMeans(n_clusters=2, random_state=self.random_state)
        km.fit(data)
        centers = km.cluster_centers_
        v = centers[0] - centers[1]
        x_prime = data.dot(v) / v.dot(v)
        is_norm = self.anderson_darling_norm_test(x_prime)

        if is_norm:
            self.data_index[index[:, 0]] = index
            return

        unique_labels = set(km.labels_)
        for k in unique_labels:
            current_data = data[km.labels_ == k]
            
            if current_data.shape[0] <= self.min_obs:
                self.data_index[index[:, 0]] = index
                return
            
            current_index = index[km.labels_==k]
            current_index[:, 1] = np.random.randint(1,100000000000)
            self._recursiveprocess(data=current_data, depth=depth, index=current_index)
    
    def fit(self, data):
        self.data = data
        data_index = np.array([[i, False] for i in range(data.shape[0])])
        self.data_index = data_index
        self._recursiveprocess(data=data, depth=0, index=data_index)
        self.labels = np.unique(self.data_index[:, 1], return_inverse=True)[1]
        
    def fit_predict(self, data):
        if hasattr(self, "labels"):
            return self.labels
        else:
            self.fit(data)
            return self.labels
