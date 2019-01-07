#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py


def SRGAN_g(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    _, nx, ny, nz = t_image.get_shape().as_list()
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
      conv = InputLayer(t_image, name='in')
      
      conv0 = Conv2d(conv, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_0') 
      conv0 = BatchNormLayer(conv0,is_train=is_train, gamma_init=gamma_init, name='bn_0')
      conv1 = Conv2d(conv0, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_1') 
      conv1 = BatchNormLayer(conv1,is_train=is_train, gamma_init=gamma_init, name='bn_1')
      conv2 = Conv2d(conv1, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_2') 
      conv2 = BatchNormLayer(conv2,is_train=is_train, gamma_init=gamma_init, name='bn_2')
      #Size is 128x128
      mp1=MaxPool2d(conv2,(2,2),name='mp_1')
      #Size is 64x64
      
      conv3 = Conv2d(mp1, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_3') 
      conv3 = BatchNormLayer(conv3,is_train=is_train, gamma_init=gamma_init, name='bn_3')
      conv4 = Conv2d(conv3, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_4') 
      conv4 = BatchNormLayer(conv4,is_train=is_train, gamma_init=gamma_init, name='bn_4')
      conv5 = Conv2d(conv4, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_5') 
      conv5 = BatchNormLayer(conv5,is_train=is_train, gamma_init=gamma_init, name='bn_5')
      mp2=MaxPool2d(conv5,(2,2),name='mp_2')
      #Size is 32x32
      
      
      conv6 = Conv2d(mp2, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_6') 
      conv6 = BatchNormLayer(conv6,is_train=is_train, gamma_init=gamma_init, name='bn_6')
      conv7 = Conv2d(conv6, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_7') 
      conv7 = BatchNormLayer(conv7,is_train=is_train, gamma_init=gamma_init, name='bn_7')
      conv8 = Conv2d(conv7, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_8') 
      conv8 = BatchNormLayer(conv8,is_train=is_train, gamma_init=gamma_init, name='bn_8')
      mp3=MaxPool2d(conv8,(2,2),name='mp_3')
      #Size is 16x16
      
      
      conv9 = Conv2d(mp3, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_9') 
      conv9 = BatchNormLayer(conv9,is_train=is_train, gamma_init=gamma_init, name='bn_9')
      conv10 = Conv2d(conv9, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_10') 
      conv10 = BatchNormLayer(conv10,is_train=is_train, gamma_init=gamma_init, name='bn_10')
      conv11 = Conv2d(conv10, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_11') 
      conv11 = BatchNormLayer(conv11,is_train=is_train, gamma_init=gamma_init, name='bn_11')
      #Size is 16x16
      
      mp4=MaxPool2d(conv11,(2,2),name='mp_4')
      #Size is 8x8
      
      deconv = mp4 
     
      #====onv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')

      deconv0 = DeConv2d(deconv, 128, (3, 3), (8,8), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_0')
      deconv0 = BatchNormLayer(deconv0,is_train=is_train, gamma_init=gamma_init, name='bn_12')
      deconv1 = DeConv2d(deconv0, 128, (3, 3), (8,8), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_1')
      deconv1 = BatchNormLayer(deconv1,is_train=is_train, gamma_init=gamma_init, name='bn_13')
      deconv2 = DeConv2d(deconv1, 128, (3, 3), (8,8), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_2')
      deconv2 = BatchNormLayer(deconv2,is_train=is_train, gamma_init=gamma_init, name='bn_14')
      #Size is 8x8
      up1=UpSampling2dLayer(deconv2,(2,2))
      #Size is 16x16
      
      deconv3 = DeConv2d(up1, 128, (3, 3), (16,16), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_3')
      deconv3 = BatchNormLayer(deconv3,is_train=is_train, gamma_init=gamma_init, name='bn_15')
      deconv4 = DeConv2d(deconv3, 128, (3, 3), (16,16), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_4')
      deconv4 = BatchNormLayer(deconv4,is_train=is_train, gamma_init=gamma_init, name='bn_16')
      deconv5 = DeConv2d(deconv4, 128, (3, 3), (16,16), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_5')
      deconv5 = BatchNormLayer(deconv5,is_train=is_train, gamma_init=gamma_init, name='bn_17')

      add2 = ElementwiseLayer([mp3, deconv5], tf.add, name='add2')
      up2=UpSampling2dLayer(add2,(2,2))
      #Size is 32x32
      
      deconv6 = DeConv2d(up2, 128, (3, 3), (32,32), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_6')
      deconv6 = BatchNormLayer(deconv6,is_train=is_train, gamma_init=gamma_init, name='bn_18')
      deconv7 = DeConv2d(deconv6, 128, (3, 3), (32,32), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_7')
      deconv7 = BatchNormLayer(deconv7,is_train=is_train, gamma_init=gamma_init, name='bn_19')
      deconv8 = DeConv2d(deconv7, 128, (3, 3), (32,32), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_8')
      deconv8 = BatchNormLayer(deconv8,is_train=is_train, gamma_init=gamma_init, name='bn_20')
      
      add3 = ElementwiseLayer([mp2, deconv8], tf.add, name='add3')
      up3=UpSampling2dLayer(add3,(2,2))
      #Size is 64x64
      
      deconv9 = DeConv2d(up3, 128, (3, 3), (64,64), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_9')
      deconv9 = BatchNormLayer(deconv9,is_train=is_train, gamma_init=gamma_init, name='bn_21')
      deconv10 = DeConv2d(deconv9, 128, (3, 3), (64,64), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_10')
      deconv10 = BatchNormLayer(deconv10,is_train=is_train, gamma_init=gamma_init, name='bn_22')
      deconv11 = DeConv2d(deconv10, 128, (3, 3), (64,64), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_11')
      deconv11 = BatchNormLayer(deconv11,is_train=is_train, gamma_init=gamma_init, name='bn_23')
      
      add4 = ElementwiseLayer([mp1, deconv11], tf.add, name='add4')
      up4=UpSampling2dLayer(add4,(2,2))
      #Size is 128x128
      
      convout = Conv2d(up4, 1, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, name='convout') 
      return convout




  #      # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
  #      conv = InputLayer(t_image, name='in')
  #      convs = []
  #      for i in range(15):
  #          conv = Conv2d(conv, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='conv_%s' % i)
  #          conv = BatchNormLayer(conv,is_train=is_train, gamma_init=gamma_init, name='bn_%s' % i)            
  #          if (i%3==0 and i!=0):
  #              conv=MaxPool2d(conv,(2,2),name='mp_%s' % i)

  #          convs.append(conv)

  #      deconv = conv

  #      for i in range(15):


  #         # if i<14 and i%2 == 0:
  #          
  #          if (i%3==0 and i!=0) :
  #             #print(nx//(2**(4-i//3)))
  #             deconv = ElementwiseLayer([convs[15 - i], deconv], tf.add, name='add_%s' % i)
  #             deconv=UpSampling2dLayer(deconv,(nx//(2**(4-i//3)),ny//(2**(4-i//3))))
  #          deconv = DeConv2d(deconv, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='deconv_%s' % i)
  #          deconv = BatchNormLayer(deconv,is_train=is_train, gamma_init=gamma_init, name='bn_%s'%(i+15))            

  #    #  n = SubpixelConv2d(deconv, scale=2, n_out_channel=None, act=tf.nn.relu, name='upsample_1')
  #    #  n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='upsample_2')
  #      n = Conv2d(deconv, 1, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init, name='out')



def SRGAN_g2(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)

    96x96 --> 384x384

    Use Resize Conv
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    size = t_image.get_shape().as_list()
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        #
        # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        ## 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        n = UpSampling2dLayer(n, size=[size[1] * 2, size[2] * 2], is_scale=False, method=1, align_corners=False, name='up1/upsample2d')
        n = Conv2d(n, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up1/conv2d')  # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up1/batch_norm')

        n = UpSampling2dLayer(n, size=[size[1] * 4, size[2] * 4], is_scale=False, method=1, align_corners=False, name='up2/upsample2d')
        n = Conv2d(n, 32, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up2/conv2d')  # <-- may need to increase n_filter
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up2/batch_norm')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d2(t_image, is_train=False, reuse=False):
    """ Discriminator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')

        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits


def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c')

        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')

        net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [400, 400, 1]
        assert green.get_shape().as_list()[1:] == [400, 400, 1]
        assert blue.get_shape().as_list()[1:] == [400, 400, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [400, 400, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv


# def vgg16_cnn_emb(t_image, reuse=False):
#     """ t_image = 244x244 [0~255] """
#     with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
#         tl.layers.set_name_reuse(reuse)
#
#         mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
#         net_in = InputLayer(t_image - mean, name='vgg_input_im')
#         """ conv1 """
#         network = tl.layers.Conv2dLayer(net_in,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool1')
#         """ conv2 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool2')
#         """ conv3 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool3')
#         """ conv4 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_3')
#
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool4')
#         conv4 = network
#
#         """ conv5 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool5')
#
#         network = FlattenLayer(network, name='vgg_flatten')
#
#         # # network = DropoutLayer(network, keep=0.6, is_fix=True, is_train=is_train, name='vgg_out/drop1')
#         # new_network = tl.layers.DenseLayer(network, n_units=4096,
#         #                     act = tf.nn.relu,
#         #                     name = 'vgg_out/dense')
#         #
#         # # new_network = DropoutLayer(new_network, keep=0.8, is_fix=True, is_train=is_train, name='vgg_out/drop2')
#         # new_network = DenseLayer(new_network, z_dim, #num_lstm_units,
#         #             b_init=None, name='vgg_out/out')
#         return conv4, network
