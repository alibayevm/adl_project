import logging
import os
from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train_sess(sess, model_spec, stats):
    total_loss = model_spec['total_loss']
    update_metrics = model_spec['update_metrics']
    train_op = model_spec['train_op']
    metrics = model_spec['metrics']

    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    t = trange(model_spec['num_steps'])
    for _ in t:
        _, _, t_loss = sess.run([train_op, update_metrics, total_loss])
        t.set_postfix(loss='{:05.3f}'.format(t_loss))
    
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)
    
    stats['vv'].append(metrics_val['loss_vv'])
    stats['tt'].append(metrics_val['loss_tt'])
    stats['vt'].append(metrics_val['loss_vt'])
    stats['tv'].append(metrics_val['loss_tv'])
    stats['total'].append(metrics_val['total_loss'])


def evaluate_sess(sess, model_spec, stats):
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']

    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    t = trange(model_spec['num_steps'])
    for _ in t:
        sess.run(update_metrics)
    
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Validation metrics: " + metrics_string)

    stats['vv'].append(metrics_val['loss_vv'])
    stats['tt'].append(metrics_val['loss_tt'])
    stats['vt'].append(metrics_val['loss_vt'])
    stats['tv'].append(metrics_val['loss_tv'])
    stats['total'].append(metrics_val['total_loss'])

    return metrics_val['total_loss']


def train_and_validate_model(train_model_spec, valid_model_spec, model_dir, params):
    """
    Trains the model and evaluates it on validation set.
    """
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)

    stats_train = {'vv': [], 'tt': [], 'vt': [], 'tv': [], 'total': []}
    stats_valid = {'vv': [], 'tt': [], 'vt': [], 'tv': [], 'total': []}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        best_valid_loss = float('inf')

        for epoch in range(params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            # Train the model
            train_sess(sess, train_model_spec, stats_train)

            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)

            # Evaluate the model
            valid_loss = evaluate_sess(sess, valid_model_spec, stats_valid)

            # Check if model is best so far
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))

            plt.clf()
            plt.plot(range(1, epoch + 2), stats_train['vv'], 'r', label='Train loss')
            plt.plot(range(1, epoch + 2), stats_valid['vv'], 'b', label='Valid loss')
            plt.title('Video to video loss')
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'vv.png'))

            plt.clf()
            plt.plot(range(1, epoch + 2), stats_train['tt'], 'r', label='Train loss')
            plt.plot(range(1, epoch + 2), stats_valid['tt'], 'b', label='Valid loss')
            plt.title('Text to text loss')
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'tt.png'))

            plt.clf()
            plt.plot(range(1, epoch + 2), stats_train['vt'], 'r', label='Train loss')
            plt.plot(range(1, epoch + 2), stats_valid['vt'], 'b', label='Valid loss')
            plt.title('Video to text loss')
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'vt.png'))

            plt.clf()
            plt.plot(range(1, epoch + 2), stats_train['tv'], 'r', label='Train loss')
            plt.plot(range(1, epoch + 2), stats_valid['tv'], 'b', label='Valid loss')
            plt.title('Text to video loss')
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'tv.png'))

            plt.clf()
            plt.plot(range(1, epoch + 2), stats_train['total'], 'r', label='Train loss')
            plt.plot(range(1, epoch + 2), stats_valid['total'], 'b', label='Valid loss')
            plt.title('Total loss')
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'total.png'))

        # Save the loss values
        # np.save(os.path.join(model_dir, 'vv.npy'), np.stack(stats['vv']), allow_pickle=False, fix_imports=False)
        # np.save(os.path.join(model_dir, 'tt.npy'), np.stack(stats['tt']), allow_pickle=False, fix_imports=False)
        # np.save(os.path.join(model_dir, 'vt.npy'), np.stack(stats['vt']), allow_pickle=False, fix_imports=False)
        # np.save(os.path.join(model_dir, 'tv.npy'), np.stack(stats['tv']), allow_pickle=False, fix_imports=False)
        # np.save(os.path.join(model_dir, 'total.npy'), np.stack(stats['total']), allow_pickle=False, fix_imports=False)