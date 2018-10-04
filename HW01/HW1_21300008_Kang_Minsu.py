import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt

plt.switch_backend('agg')

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    return train_set, valid_set, test_set



if __name__ == '__main__':

    img_width, img_height = 28, 28

    train_set, _, _ = load_data('mnist.pkl.gz')

    train_x, _ = train_set

    reshaped_train_x = np.reshape(train_x, (-1, img_width, img_height))

    mean = np.mean(reshaped_train_x, axis=0)
    scipy.misc.imsave('mean.jpg', mean)
    # generate image of mean

    var = np.var(reshaped_train_x, axis = 0)
    scipy.misc.imsave('var.jpg', var)

    cov_matrix = np.cov(train_x.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)    
    eigenvectors = eigenvectors.T
 
    for i in range(10):
       scipy.misc.imsave('eigenvector{:02d}'.format(i+1)+'.jpg', np.reshape(eigenvectors[i], (img_width, img_height)))

    plt.xlabel('eigenvalue number')
    plt.ylabel('eigenvalue')
    plt.plot(range(100), eigenvalues[:100].tolist())
    plt.savefig('plot_eigenvalue.jpg')

