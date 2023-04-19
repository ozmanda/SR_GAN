import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def conv_layer_2d(x, filter_shape, stride, trainable=True):
    W = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.compat.v1.contrib.layers.xavier_initializer(),
        trainable=trainable)
    b = tf.compat.v1.get_variable(
        name='bias',
        shape=[filter_shape[-1]],
        dtype=tf.float32,
        initializer=tf.compat.v1.contrib.layers.xavier_initializer(),
        trainable=trainable)
    x = tf.nn.bias_add(tf.compat.v1.nn.conv2d(
        input=x,
        filter=W,
        strides=[1, stride, stride, 1],
        padding='SAME'), b)

    return x

def deconv_layer_2d(x, filter_shape, output_shape, stride, trainable=True):
    # according to https://stackoverflow.com/questions/64255154/change-tf-contrib-layers-xavier-initializer-to-2-0-0,
    # tf.contrib.layers.xavier_initializer() has been changed to tf.keras.initializers.GlorotUniform()
    x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], mode='reflect')
    W = tf.compat.v1.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.keras.initializers.GlorotUniform(),
        trainable=trainable)
    b = tf.compat.v1.get_variable(
        name='bias',
        shape=[output_shape[-1]],
        dtype=tf.float32,
        initializer=tf.keras.initializers.GlorotUniform(),
        trainable=trainable)
    x = tf.nn.bias_add(tf.compat.v1.nn.conv2d_transpose(
        value=x,
        filter=W,
        output_shape=output_shape,
        strides=[1, stride, stride, 1],
        padding='SAME'), b)

    return x[:, 3:-3, 3:-3, :]

def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    x = tf.reshape(transposed, [-1, dim])

    return x

def dense_layer(x, out_dim, trainable=True):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.compat.v1.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
        trainable=trainable)
    b = tf.compat.v1.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    x = tf.add(tf.matmul(x, W), b)

    return x

def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        N, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, (N, h, w, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, h, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, w, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.reshape(x, (N, h*r, w*r, 1))

    xc = tf.split(x, n_split, 3)
    x = tf.concat([PS(x_, r) for x_ in xc], 3)

    return x

def plot_SR_data(idx, LR, SR, path):

    for i in range(LR.shape[0]):
        vmin0, vmax0 = np.min(SR[i,:,:,0]), np.max(SR[i,:,:,0])
        vmin1, vmax1 = np.min(SR[i,:,:,1]), np.max(SR[i,:,:,1])

        plt.figure(figsize=(12, 12))
        
        plt.subplot(221)
        plt.imshow(LR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
        plt.title('LR 0 Input', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(223)
        plt.imshow(LR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
        plt.title('LR 1 Input', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(222)
        plt.imshow(SR[i, :, :, 0], vmin=vmin0, vmax=vmax0, cmap='viridis', origin='lower')
        plt.title('SR 0 Output', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])
        
        plt.subplot(224)
        plt.imshow(SR[i, :, :, 1], vmin=vmin1, vmax=vmax1, cmap='viridis', origin='lower')
        plt.title('SR 1 Output', fontsize=9)
        plt.colorbar()
        plt.xticks([], [])
        plt.yticks([], [])

        plt.savefig(path+'/img{0:05d}.png'.format(idx[i]), dpi=200, bbox_inches='tight')
        plt.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def downscale_image(x, K):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    if x.ndim == 3:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    if x.ndim == 2:
        x = x.reshape((1, 1, x.shape[0], x.shape[1]))

    x_in = tf.compat.v1.placeholder(tf.float64, [None, x.shape[1], x.shape[2], x.shape[3]])

    weight = tf.constant(1.0/K**2, shape=[K, K, x.shape[3], x.shape[3]], dtype=tf.float64)
    downscaled = tf.compat.v1.nn.conv2d(x_in, filter=weight, strides=[1, K, K, 1], padding='SAME')

    with tf.compat.v1.Session() as sess:
        ds_out = sess.run(downscaled, feed_dict={x_in: x})

    return ds_out

def generate_TFRecords(filename, data_HR=None, data_LR=None, mode='test'):
    '''
        Generate TFRecords files for model training or testing

        inputs:
            filename - filename for TFRecord (should by type *.tfrecord)
            data     - numpy array of size (N, h, w, c) containing data to be written to TFRecord
            model    - if 'train', then data contains HR data that is coarsened k times 
                       and both HR and LR data written to TFRecord
                       if 'test', then data contains LR data 
            K        - downscaling factor, must be specified in training mode

        outputs:
            No output, but .tfrecord file written to filename
    '''
    if mode == 'train':
        assert data_HR is not None, 'In training mode, high resolution data must be given'
        assert data_LR is not None, 'In training mode, low resolution data must be given'
        shape = data_HR.shape[0]
    if mode == 'test':
        assert data_LR is not None, 'In testing mode, low-resolution data must be given'
        shape = data_LR.shape[0]

    with tf.compat.v1.python_io.TFRecordWriter(filename) as writer:
        for j in range(shape):
            if mode == 'train':
                h_HR, w_HR, c = data_HR[j, ...].shape
                h_LR, w_LR, c = data_LR[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                   'data_HR': _bytes_feature(data_HR[j, ...].tostring()),
                                      'h_HR': _int64_feature(h_HR),
                                      'w_HR': _int64_feature(w_HR),
                                         'c': _int64_feature(c)})
            elif mode == 'test':
                h_LR, w_LR, c = data_LR[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                         'c': _int64_feature(c)})

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def calculate_mu_sig(pickledata):
    # loop through images
    shp = pickledata.shape
    count = pickledata.size
    total_sum = 0
    total_sqsum = 0

    for imgidx in range(pickledata.shape[0]):
        total_sum += np.sum(pickledata[imgidx, ...], axis=(0, 1))
        total_sqsum += np.sum(pickledata[imgidx, ...]**2, axis=(0, 1))

    mu = total_sum / count
    sigma = np.sqrt(total_sqsum - total_sum**2 / count)

    return [mu, sigma]




