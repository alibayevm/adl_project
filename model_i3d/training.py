import logging
import os
from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_i3d.utils import save_dict_to_json


def train_model(train_model, valid_model, model_dir, params):
    """
    Trains the model
    """
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)

    restore = tf.train.Saver(var_list=train_model['variable_map'], reshape=True)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore.restore(sess, params.restore_path)

        best_acc = 0.0
        best_loss = float('inf')
        best_metric = float('inf')
        

        for epoch in range(params.num_epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            
            # ===================================== Training ==============================================
            lr = params.lr * params.lr_decline ** (epoch // params.change_lr)
            lr_holder = train_model['lr']
            
            sess.run(train_model['iterator_init_op'])
            sess.run(train_model['metrics_init_op'])

            update_metrics_op = train_model['update_metrics_op']
            train_op = train_model['train_op'] if epoch >= params.first_epochs else train_model['train_op_initial']

            for _ in trange(train_model['num_steps']):
                sess.run([train_op, update_metrics_op], feed_dict={lr_holder: lr})
            
            metrics_values = {k: v[0] for k, v in train_model['metrics'].items()}
            metrics_val = sess.run(metrics_values)
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
            logging.info("- Train metrics: " + metrics_string)
            loss = metrics_val['loss']

            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)


            # ===================================== Evaluation ==============================================
            sess.run(valid_model['iterator_init_op'])
            sess.run(valid_model['metrics_init_op'])
            conf_mat = np.zeros((26, 26), dtype=np.int64)

            update_metrics_op = valid_model['update_metrics_op']

            for _ in trange(valid_model['num_steps']):
                _, conf_mat_batch = sess.run([update_metrics_op, valid_model['conf_mat']])
                conf_mat += conf_mat_batch
            
            metrics_values = {k: v[0] for k, v in valid_model['metrics'].items()}
            metrics_val = sess.run(metrics_values)
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
            logging.info("- Validation metrics: " + metrics_string)
            accuracy = metrics_val['accuracy']

            if best_acc < accuracy:
                conf_mat_acc = conf_mat
                best_acc = accuracy

                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy, saving in {}".format(best_save_path))

                save_dict_to_json({'accuracy': accuracy}, os.path.join(model_dir, 'best_acc.json'))
            
            if best_loss > loss:
                conf_mat_loss = conf_mat
                best_loss = loss

                save_dict_to_json({'loss': loss}, os.path.join(model_dir, 'best_loss.json'))
            
            if best_metric > (loss + 1 - accuracy):
                conf_mat_best = conf_mat
                best_metric = (loss + 1 - accuracy)

                save_dict_to_json({'accuracy': accuracy, 'loss': loss}, os.path.join(model_dir, 'best_metrics.json'))
        
        np.savez('conf_mat.npz', conf_mat_acc=conf_mat_acc, conf_mat_loss=conf_mat_loss, conf_mat_best=conf_mat_best)