"""
直接用 pointnet-master 原版语义分割结构训练焊缝二类（缝/背景）。
数据由 seam_localization 导出：先运行 export_seam_to_pointnet_h5.py 生成 sem_seg_seam_data/*.h5。
"""
import argparse
import h5py
import numpy as np
import tensorflow as tf
import os
import sys
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model_seam import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log_dir', default='log_seam', help='Log dir')
parser.add_argument('--num_point', type=int, default=2048)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default=None, help='sem_seg_seam_data 目录，默认 ROOT/sem_seg_seam_data')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
LOG_DIR = FLAGS.log_dir
NUM_CLASSES = 2

if FLAGS.data_dir is None:
    FLAGS.data_dir = os.path.join(ROOT_DIR, 'sem_seg_seam_data')
train_h5 = os.path.join(FLAGS.data_dir, 'seam_train.h5')
val_h5 = os.path.join(FLAGS.data_dir, 'seam_val.h5')
if not os.path.isfile(train_h5) or not os.path.isfile(val_h5):
    print('未找到焊缝 h5 数据。请先运行:')
    print('  python seam_localization/pointcloud_dataset.py --num_train 200 --num_val 50')
    print('  python seam_localization/export_seam_to_pointnet_h5.py')
    sys.exit(1)

DECAY_STEP = 20000
DECAY_RATE = 0.5
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

os.makedirs(LOG_DIR, exist_ok=True)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

# 加载数据：与 pointnet sem_seg 相同格式 data (N,P,9), label (N,P)
with h5py.File(train_h5, 'r') as f:
    train_data = f['data'][:].astype(np.float32)
    train_label = f['label'][:].astype(np.int32)
with h5py.File(val_h5, 'r') as f:
    test_data = f['data'][:].astype(np.float32)
    test_label = f['label'][:].astype(np.int32)
print('train_data', train_data.shape, 'train_label', train_label.shape)
print('test_data', test_data.shape, 'test_label', test_label.shape)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    lr = tf.train.exponential_decay(
        BASE_LEARNING_RATE, batch * BATCH_SIZE, DECAY_STEP, DECAY_RATE, staircase=True)
    return tf.maximum(lr, 1e-5)


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY, batch * BATCH_SIZE, BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE, staircase=True)
    return tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.cast(BATCH_SIZE * NUM_POINT, tf.float32)
            learning_rate = get_learning_rate(batch)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer(), {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl, 'labels_pl': labels_pl,
               'is_training_pl': is_training_pl, 'pred': pred, 'loss': loss,
               'train_op': train_op, 'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % epoch)
            # train
            current_data, current_label, _ = provider.shuffle_data(
                train_data[:, 0:NUM_POINT, :], train_label)
            num_batches = current_data.shape[0] // BATCH_SIZE
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: True}
                _, step, loss_val, pred_val = sess.run(
                    [ops['train_op'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
                pred_val = np.argmax(pred_val, 2)
                total_correct += np.sum(pred_val == current_label[start_idx:end_idx])
                total_seen += BATCH_SIZE * NUM_POINT
                loss_sum += loss_val
            log_string('train loss: %f accuracy: %f' % (loss_sum / num_batches, total_correct / float(total_seen)))

            # eval
            num_batches_test = test_data.shape[0] // BATCH_SIZE
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            for batch_idx in range(num_batches_test):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                feed_dict = {ops['pointclouds_pl']: test_data[start_idx:end_idx, :, :],
                             ops['labels_pl']: test_label[start_idx:end_idx],
                             ops['is_training_pl']: False}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                pred_val = np.argmax(pred_val, 2)
                total_correct += np.sum(pred_val == test_label[start_idx:end_idx])
                total_seen += BATCH_SIZE * NUM_POINT
                loss_sum += loss_val
            log_string('eval loss: %f accuracy: %f' % (loss_sum / num_batches_test, total_correct / float(total_seen)))

            if (epoch + 1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
                log_string('Model saved: %s' % save_path)

    LOG_FOUT.close()


if __name__ == '__main__':
    train()
