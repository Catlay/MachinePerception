from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from six.moves import xrange # pylint: disable=redefined-builtin
import copy

class Skeleton(object):
    """
    Represents a skeleton, i.e. defines how the joints are linked together and the offset for each bone.
    """
    end_effectors = np.array([5, 10, 15, 21, 23, 29, 31])

    parents = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                        17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    n_joints = len(parents)

    offsets = np.array([[0.000000, 0.000000, 0.000000],
                        [-132.948591, 0.000000, 0.000000],
                        [0.000000, -442.894612, 0.000000],
                        [0.000000, -454.206447, 0.000000],
                        [0.000000, 0.000000, 162.767078],
                        [0.000000, 0.000000, 74.999437],
                        [132.948826, 0.000000, 0.000000],
                        [0.000000, -442.894413, 0.000000],
                        [0.000000, -454.206590, 0.000000],
                        [0.000000, 0.000000, 162.767426],
                        [0.000000, 0.000000, 74.999948],
                        [0.000000, 0.100000, 0.000000],
                        [0.000000, 233.383263, 0.000000],
                        [0.000000, 257.077681, 0.000000],
                        [0.000000, 121.134938, 0.000000],
                        [0.000000, 115.002227, 0.000000],
                        [0.000000, 257.077681, 0.000000],
                        [0.000000, 151.034226, 0.000000],
                        [0.000000, 278.882773, 0.000000],
                        [0.000000, 251.733451, 0.000000],
                        [0.000000, 0.000000, 0.000000],
                        [0.000000, 0.000000, 99.999627],
                        [0.000000, 100.000188, 0.000000],
                        [0.000000, 0.000000, 0.000000],
                        [0.000000, 257.077681, 0.000000],
                        [0.000000, 151.031437, 0.000000],
                        [0.000000, 278.892924, 0.000000],
                        [0.000000, 251.728680, 0.000000],
                        [0.000000, 0.000000, 0.000000],
                        [0.000000, 0.000000, 99.999888],
                        [0.000000, 137.499922, 0.000000],
                        [0.000000, 0.000000, 0.000000]])


def exp2rotmat(expmap):
    """
    Converts joint angles in exponential map format to 3-by-3 rotation matrices. This is an implementation of the
    Rodrigues formula.
    :param expmap: np array of shape (N, 3)
    :return: np array of shape (N, 3, 3)
    """
    theta = np.linalg.norm(expmap, axis=-1, keepdims=True)
    axis = expmap / (theta + np.finfo(np.float32).eps)
    cost = np.cos(theta)[..., np.newaxis]
    sint = np.sin(theta)[..., np.newaxis]
    rrt = np.matmul(axis[..., np.newaxis], axis[:, np.newaxis, ...])
    skew = np.zeros([expmap.shape[0], 3, 3])
    skew[:, 0, 1] = -axis[:, 2]
    skew[:, 0, 2] = axis[:, 1]
    skew[:, 1, 2] = -axis[:, 0]
    skew = skew - np.transpose(skew, [0, 2, 1])
    R = np.repeat(np.eye(3, 3)[np.newaxis, ...], expmap.shape[0], axis=0)
    R = R*cost + (1 - cost)*rrt + sint*skew
    return R


def forward_kinematics(expmap, skeleton=Skeleton, root_pos=None):
    """
    Compute joint positions from joint angles represented as exponential maps.
    :param expmap: np array of shape (seq_length, n_joints*3)
    :param skeleton: skeleton defining parents and offsets per joint
    :param root_pos: root positions per frame as an np array of shape (seq_length, 3) or None
    :return: np array of shape (seq_length, n_joints, 3)
    """
    seq_length = expmap.shape[0]
    n_joints = expmap.shape[1] // 3

    roots = np.zeros([seq_length, 3]) if root_pos is None else root_pos
    angles = np.reshape(expmap, [seq_length, n_joints, 3])

    if n_joints < skeleton.n_joints:
        # the input misses the end effectors, so insert zero vectors at their place
        non_end_effectors = [j for j in range(skeleton.n_joints) if j not in skeleton.end_effectors]
        angles_corr = np.zeros([seq_length, skeleton.n_joints, 3])
        angles_corr[:, non_end_effectors] = angles
        angles = angles_corr
        n_joints = angles.shape[1]

    assert roots.shape[0] == seq_length, 'must have as many root positions as there are frames'
    assert n_joints == skeleton.n_joints, 'unexpected number of joints in skeleton'

    positions = np.zeros_like(angles)  # output positions
    rotations = np.zeros([n_joints, 3, 3])  # must temporarily save rotation matrices for each frame

    # precompute rotation matrices for speedup
    rotmat = exp2rotmat(np.reshape(angles, [-1, 3]))
    rotmat = np.reshape(rotmat, [seq_length, n_joints, 3, 3])

    for f in range(seq_length):
        for j in range(n_joints):

            if skeleton.parents[j] == -1:  # this is the root
                positions[f, j] = skeleton.offsets[j] + roots[f, j]
                rotations[j] = rotmat[f, j]

            else:  # this is a regular joint
                positions[f, j] = np.matmul(skeleton.offsets[j:j+1], rotations[skeleton.parents[j]])
                positions[f, j] += positions[f, skeleton.parents[j]]
                rotations[j] = np.matmul(rotmat[f, j], rotations[skeleton.parents[j]])

    positions = positions[:, :, [0, 2, 1]]  # swap y and z axis
    positions = np.reshape(positions, [seq_length, -1])
    return positions


def padded_array_to_list(data, mask):
    """
    Converts a padded numpy array to a list of un-padded numpy arrays. `data` is expected in shape
    (n, max_seq_length, ...) and `mask` in shape (n, max_seq_length). The returned value is a list of size n, each
    element being an np array of shape (dynamic_seq_length, ...).
    """
    converted = []
    seq_lengths = np.array(np.sum(mask, axis=1), dtype=np.int)
    for i in range(data.shape[0]):
        converted.append(data[i, 0:seq_lengths[i], ...])
    return converted


def export_config(config, output_file):
    """
    Write the configuration parameters into a human readable file.
    :param config: the configuration dictionary
    :param output_file: the output text file
    """
    if not output_file.endswith('.txt'):
        output_file.append('.txt')
    max_key_length = np.amax([len(k) for k in config.keys()])
    with open(output_file, 'w') as f:
        for k in sorted(config.keys()):
            out_string = '{:<{width}}: {}\n'.format(k, config[k], width=max_key_length)
            f.write(out_string)


def export_to_csv(data, ids, output_file):
    """
    Write an array into a csv file.
    :param data: np array of shape (n, seq_length, dof)
    :param ids: array of size n specifying an id for each respective entry in data
    :param output_file: where to store the data
    """
    # treat every frame as a separate sample, otherwise Kaggle goes berserk
    n_samples, seq_length, dof = data.shape
    data_r = np.reshape(data, [-1, dof])

    ids_per_frame = [['{}_{}'.format(id_, i) for i in range(seq_length)] for id_ in ids]
    ids_per_frame = np.reshape(np.array(ids_per_frame), [-1])

    data_frame = pd.DataFrame(data_r,
                              index=ids_per_frame,
                              columns=['dof{}'.format(i) for i in range(dof)])
    data_frame.index.name = 'Id'

    if not output_file.endswith('.csv'):
        output_file.append('.csv')

    data_frame.to_csv(output_file, float_format='%.8f')

def one_hot_encoding( data, labels):
  """
  add features with one-hot encoding of activity/label
  """
  
  n_labels = 14
  data_onehot = []

  for d in range(0,len(data)):    
    onehot = np.zeros((data[d].shape(0), n_labels))        # one-hot matrix with zeroes
    label = int(data[d]['action_label'])
    onehot[label] = 1                                      # set target idx to 1
    data_onehot[d] = np.append(data[d],onehot, axis=1)     # add one_hot vector to data.
    #print('data onehot: ', data_onehot)
  
  return data_onehot


def standardize_data( data, data_mean, data_std, dim_to_use, one_hot ):
  """
  Standardize input data by getting rid of unused dimensions, subtracting the mean and
  dividing by the standard deviation

  Args
    data: list of numpy arrays with data to normalize
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dim_to_use: vector with dimensions used by the model
  Returns
    data_out: the passed data matrix, but normalized
  """
  data_out = data

    # No one-hot encoding... no need to do anything special
  for s in range(0,len(data)):
    data_out[s] = np.divide( (data[s]-data_mean), data_std )
    data_out[s] = data_out[s][ :, dim_to_use ]

  return data_out


def standardization_stats(completeData):
  """"  RAH: Adapted from Martinez.
  Args
    completeData: list of arrays with data to standardize
  Returns
    data_mean: vector of mean used to standardize the data
    data_std: vector of standard deviation used to standardize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
  """
  
  data_reshaped = completeData[0]
  for s in range (1, len(completeData)): 
    data_reshaped = np.concatenate((data_reshaped , completeData[s])) 

  data_mean = np.mean ( data_reshaped, axis=(0) ) # mean over all samples of all the 75 features
  data_std  = np.std ( data_reshaped, axis=(0) ) # std over all samples of all the 75 features 
  
  dimensions_to_ignore = []
  dimensions_to_use    = []

  dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
  dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )

  data_std[dimensions_to_ignore] = 1.0

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use

def unStandardizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, one_hot ):
  """  Adapted from Martinez:
  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  dimensions_to_use = np.array(dimensions_to_use)

  if one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-14] # 14 is number of actions
  else:
    origData[:, dimensions_to_use] = normalizedData

  # potentially ineficient, but only done once per experiment
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData
