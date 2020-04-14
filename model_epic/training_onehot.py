import logging
import os
from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train_sess(sess, model_spec, stats):
    update_metrics = model_spec['update_metrics']
    train_op = model_spec['train_op']
    metrics = model_spec['metrics']

    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    t = trange(model_spec['num_steps'])
    for _ in t:
        _, _ = sess.run([train_op, update_metrics])
    
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)
    
    stats['loss'].append(metrics_val['loss'])
    stats['accuracy'].append(metrics_val['accuracy'])


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

    stats['loss'].append(metrics_val['loss'])
    stats['accuracy'].append(metrics_val['accuracy'])

    return metrics_val['accuracy']


def train_and_validate_model(train_model_spec, valid_model_spec, model_dir, params):
    """
    Trains the model and evaluates it on validation set.
    """
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)

    stats_train = {'accuracy': [], 'loss': []}
    stats_valid = {'accuracy': [], 'loss': []}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        best_valid_acc = 0.0

        for epoch in range(params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            # Train the model
            train_sess(sess, train_model_spec, stats_train)

            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)

            # Evaluate the model
            valid_acc = evaluate_sess(sess, valid_model_spec, stats_valid)

            # Check if model is best so far
            if valid_acc < best_valid_acc:
                best_valid_acc = valid_acc
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))

            plt.clf()
            plt.plot(range(1, epoch + 2), stats_train['accuracy'], 'r', label='Train accuracy')
            plt.plot(range(1, epoch + 2), stats_valid['accuracy'], 'b', label='Valid accuracy')
            plt.title('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'accuracy.png'))

            plt.clf()
            plt.plot(range(1, epoch + 2), stats_train['loss'], 'r', label='Train loss')
            plt.plot(range(1, epoch + 2), stats_valid['loss'], 'b', label='Valid loss')
            plt.title('Loss')
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'loss.png'))
