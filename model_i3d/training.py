import logging
import os
from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def train_model(model_spec, model_dir, params):
    """
    Trains the model
    """
    last_saver = tf.train.Saver(max_to_keep=1)
    restore_rgb = tf.train.Saver(var_list=model_spec['variable_map_rgb'], reshape=True)
    restore_flow = tf.train.Saver(var_list=model_spec['variable_map_flow'], reshape=True)

    stats = {'vv': [], 'tt': [], 'vt': [], 'tv': [], 'total': []}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model_spec['iterator_init_op'])

        restore_rgb.restore(sess, params.restore_path_rgb)
        restore_flow.restore(sess, params.restore_path_flow)

        loss_vv = model_spec['loss_vv']
        loss_tt = model_spec['loss_tt']
        loss_vt = model_spec['loss_vt']
        loss_tv = model_spec['loss_tv']
        total_loss = model_spec['total_loss']

        train_op = model_spec['train_op']

        t = trange(params.num_total_steps)

        for step in t:
            _, total, vv, tt, vt, tv = sess.run([train_op, total_loss, loss_vv, loss_tt, loss_vt, loss_tv])
            t.set_postfix(loss='{:05.3f}'.format(total))
            stats['vv'].append(vv)
            stats['tt'].append(tt)
            stats['vt'].append(vt)
            stats['tv'].append(tv)
            stats['total'].append(total)

            plt.clf()
            plt.plot(range(1, step + 2), stats['vv'], 'b')
            plt.title('Video to video loss')
            plt.savefig(os.path.join(model_dir, 'vv.png'))

            plt.clf()
            plt.plot(range(1, step + 2), stats['vt'], 'b')
            plt.title('Video to text loss')
            plt.savefig(os.path.join(model_dir, 'vt.png'))

            plt.clf()
            plt.plot(range(1, step + 2), stats['tv'], 'b')
            plt.title('Text to video loss')
            plt.savefig(os.path.join(model_dir, 'tv.png'))

            plt.clf()
            plt.plot(range(1, step + 2), stats['tt'], 'b')
            plt.title('Text to text loss')
            plt.savefig(os.path.join(model_dir, 'tt.png'))

            plt.clf()
            plt.plot(range(1, step + 2), stats['total'], 'b')
            plt.title('Total loss')
            plt.savefig(os.path.join(model_dir, 'total.png'))
        
        # Save the weights
        save_path = os.path.join(model_dir, 'weights')
        last_saver.save(sess, save_path)

        # Save the loss values
        np.save(os.path.join(model_dir, 'vv.npy'), np.stack(stats['vv']), allow_pickle=False, fix_imports=False)
        np.save(os.path.join(model_dir, 'tt.npy'), np.stack(stats['tt']), allow_pickle=False, fix_imports=False)
        np.save(os.path.join(model_dir, 'vt.npy'), np.stack(stats['vt']), allow_pickle=False, fix_imports=False)
        np.save(os.path.join(model_dir, 'tv.npy'), np.stack(stats['tv']), allow_pickle=False, fix_imports=False)
        np.save(os.path.join(model_dir, 'total.npy'), np.stack(stats['total']), allow_pickle=False, fix_imports=False)