import numpy as np
import pickle
import gzip
import six
import os

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

     train_x, train_y = train_set
     validation_x, validation_y = valid_set
     test_x, test_y = test_set

     # train_x = np.append(validation_x, train_x, axis = 0)
     # train_y = np.append(validation_y, train_y, axis = 0)


     return (train_x, train_y, test_x, test_y)

