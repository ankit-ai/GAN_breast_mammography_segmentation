import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np

#Source: https://github.com/zsdonghao/u-net-brain-tumor/blob/master/model.py

def u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
    return conv1


def _batch_normalization(input_tensor, is_train,gamma_init,name):
        return tf.layers.batch_normalization(input_tensor, training=is_train,name=name)

'''
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')

        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')


        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn4')

        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')


        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')

        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn6')

        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn8')

        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')

        conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init, name='bn9')

        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init, name='bn10')
        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')

        up4 = BatchNormLayer(up4, is_train=is_train, gamma_init=gamma_init, name='bn11')

        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn12')

        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn13')

        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')

        up3 = BatchNormLayer(up3, is_train=is_train, gamma_init=gamma_init, name='bn14')

        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')

        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn15')

        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')

        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn16')

        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')

        up2 = BatchNormLayer(up2, is_train=is_train, gamma_init=gamma_init, name='bn17')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')


        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn18')

        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')

        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn19')

        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')

        up1 = BatchNormLayer(up1, is_train=is_train, gamma_init=gamma_init, name='bn20')

        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')


        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn21')

        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')

        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn22')

        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
'''


def u_net_bn(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()


    gamma_init=tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("SRGAN_g", reuse=reuse):
        tl.layers.set_name_reuse(reuse)


        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')

        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')


        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn4')

        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')


        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')

        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn6')

        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn8')

        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')

        conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init, name='bn9')

        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init, name='bn10')
        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')

        up4 = BatchNormLayer(up4, is_train=is_train, gamma_init=gamma_init, name='bn11')

        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn12')

        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')

        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn13')

        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')

        up3 = BatchNormLayer(up3, is_train=is_train, gamma_init=gamma_init, name='bn14')

        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')

        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn15')

        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')

        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn16')

        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')

        up2 = BatchNormLayer(up2, is_train=is_train, gamma_init=gamma_init, name='bn17')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')


        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn18')

        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')

        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn19')

        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')

        up1 = BatchNormLayer(up1, is_train=is_train, gamma_init=gamma_init, name='bn20')

        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')


        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn21')

        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')

        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init, name='bn22')

        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')




        '''
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = _batch_normalization(conv1.outputs, is_train=is_train, gamma_init=gamma_init, name='bn1')
        conv1 = InputLayer(conv1, name='bn1_fix')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        conv1 = _batch_normalization(conv1.outputs,is_train=is_train, gamma_init=gamma_init, name='bn2')
        conv1 = InputLayer(conv1, name='bn2_fix')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')

        conv2 = _batch_normalization(conv2.outputs,is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv2 = InputLayer(conv2, name='bn3_fix')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')


        conv2 = _batch_normalization(conv2.outputs,is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv2 = InputLayer(conv2, name='bn4_fix')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')


        conv3 = _batch_normalization(conv3.outputs,is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv3 = InputLayer(conv3, name='bn5_fix')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')

        conv3 = _batch_normalization(conv3.outputs,is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv3 = InputLayer(conv3, name='bn6_fix')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')

        conv4 = _batch_normalization(conv4.outputs,is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv4 = InputLayer(conv4, name='bn7_fix')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')

        conv4 = _batch_normalization(conv4.outputs,is_train=is_train, gamma_init=gamma_init, name='bn8')

        conv4 = InputLayer(conv4, name='bn8_fix')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')

        conv5 = _batch_normalization(conv5.outputs,is_train=is_train, gamma_init=gamma_init, name='bn9')

        conv5 = InputLayer(conv5, name='bn9_fix')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        conv5 = _batch_normalization(conv5.outputs,is_train=is_train, gamma_init=gamma_init, name='bn10')

        conv5 = InputLayer(conv5, name='bn10_fix')

        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')

        up4 = _batch_normalization(up4.outputs,is_train=is_train, gamma_init=gamma_init, name='bn11')

        up4 = InputLayer(up4, name='bn11_fix')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')

        conv4 = _batch_normalization(conv4.outputs,is_train=is_train, gamma_init=gamma_init, name='bn12')

        conv4 = InputLayer(conv4, name='bn12_fix')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')

        conv4 = _batch_normalization(conv4.outputs,is_train=is_train, gamma_init=gamma_init, name='bn13')

        conv4 = InputLayer(conv4, name='bn13_fix')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')

        up3 = _batch_normalization(up3.outputs,is_train=is_train, gamma_init=gamma_init, name='bn14')

        up3 = InputLayer(up3, name='bn14_fix')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')

        conv3 = _batch_normalization(conv3.outputs,is_train=is_train, gamma_init=gamma_init, name='bn15')

        conv3 = InputLayer(conv3, name='bn15_fix')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')

        conv3 = _batch_normalization(conv3.outputs,is_train=is_train, gamma_init=gamma_init, name='bn16')

        conv3 = InputLayer(conv3, name='bn16_fix')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')

        up2 = _batch_normalization(up2.outputs,is_train=is_train, gamma_init=gamma_init, name='bn17')


        up2 = InputLayer(up2, name='bn17_fix')

        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')


        conv2 = _batch_normalization(conv2.outputs,is_train=is_train, gamma_init=gamma_init, name='bn18')

        conv2 = InputLayer(conv2, name='bn18_fix')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')

        conv2 = _batch_normalization(conv2.outputs,is_train=is_train, gamma_init=gamma_init, name='bn19')

        conv2 = InputLayer(conv2, name='bn19_fix')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')

        up1 = _batch_normalization(up1.outputs,is_train=is_train, gamma_init=gamma_init, name='bn20')

        up1 = InputLayer(up1, name='bn20_fix')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')


        conv1 = _batch_normalization(conv1.outputs,is_train=is_train, gamma_init=gamma_init, name='bn21')

        conv1 = InputLayer(conv1, name='bn21_fix')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')

        conv1 = _batch_normalization(conv1.outputs,is_train=is_train, gamma_init=gamma_init, name='bn22')

        conv1 = InputLayer(conv1, name='bn22_fix')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
        '''

    return conv1


def u_net_bn1(x, is_train=False, reuse=False, batch_size=None, pad='SAME', n_out=1):
    """image to image translation via conditional adversarial learning"""
    nx = int(x._shape[1])
    ny = int(x._shape[2])
    nz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nx, ny, nz))

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x,0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x,0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x,0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x,0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6,is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x,0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 512, (4, 4), out_size=(2, 2), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], 3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(4, 4), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], 3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(8, 8), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5] ,3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(15, 15), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(30, 30), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(60, 60), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(120, 120), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(240, 240), strides=(2, 2),
                                    padding=pad, act=tf.nn.relu, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')

        print(" * Output: %s" % out.outputs)
        # exit()

    return out
