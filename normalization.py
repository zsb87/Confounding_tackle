import numpy as np   


def standard_normalizer(x):
    x = x.T
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]   

    # create standard normalizer function based on input data statistics
    normalizer = lambda data: ((data.T - x_means)/x_stds).T
    
    # return normalizer and inverse_normalizer
    return normalizer


if __name__ == '__main__':


    x =np.array([[1, 2, 3],[3, 4, 6]])

    print(x)

    # return normalization functions based on input x
    normalizer = standard_normalizer(x)
    # normalize input by subtracting off mean and dividing by standard deviation
    x = normalizer(x)

    print(x)

    x_test =np.array([[0, 2, 1],[3, 4, 6]])

    x_test = normalizer(x_test)

    print(x_test)


