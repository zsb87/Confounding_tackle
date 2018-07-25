from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


"""
1. normalization
"""



# def standard_normalizer(x):
#     x = x.T
#     # compute the mean and standard deviation of the input
#     x_means = np.mean(x,axis = 1)[:,np.newaxis]
#     x_stds = np.std(x,axis = 1)[:,np.newaxis]   

#     # create standard normalizer function based on input data statistics
#     normalizer = lambda data: ((data.T - x_means)/x_stds).T
    
#     # return normalizer and inverse_normalizer
#     return normalizer



# # return normalization functions based on input x
# normalizer = standard_normalizer(x)
# # normalize input by subtracting off mean and dividing by standard deviation
# x = normalizer(x)

# x_test = normalizer(x_test)


"""
2. feature selection
"""



"""
3. DBSCAN
"""
db = DBSCAN(eps=0.3, min_samples=10).fit(X)