# pointnet-master/sem_seg 的焊缝二类版：输入 BxNx9，输出 BxNx2（缝=1 / 背景=0）
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

NUM_CLASSES = 2  # 缝 / 背景

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,9], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
    net = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7')
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, NUM_CLASSES, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv8')
    net = tf.squeeze(net, [2])
    return net

def get_loss(pred, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)
