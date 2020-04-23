import tensorflow as tf
import numpy as np
import os
import argparse
import logging
from tqdm import trange

from model_i3d.utils import Params
from model_i3d.utils import set_logger
from model_i3d.utils import data_info
from model_i3d.input_fn import input_fn
from model_i3d.i3d import InceptionI3d
from model_i3d.accuracy import compute_probabilities, compute_predictions


parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Path to the directory with `params.json` file')
args = parser.parse_args()


def check_dataset(inputs):
    with tf.Session() as sess:
        sess.run(inputs['iterator_init_op'])
        for _ in range(5):
            print(sess.run(inputs['clips']).shape)


if __name__ == "__main__":
    tf.set_random_seed(2020)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    model_dir = args.model_dir

    # Set parameters
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(model_dir, 'test.log'))

    # Input data pipeline
    logging.info("Creating the dataset...")

    test_data = data_info('test')
    test_inputs = input_fn(test_data, params, is_training=False)

    # ====================================== Define the model =======================================
    logging.info("Building the model")

    clips = test_inputs['clips']
    labels = test_inputs['labels']
    words = test_inputs['words']

    with tf.variable_scope('Model'):
        # Define the RGB model
        with tf.variable_scope('RGB'):
            model = InceptionI3d(num_classes=1024)
            logits_visual, _ = model(clips, is_training=False, dropout_keep_prob=1.0)

        # Embedding step
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)
        logits_visual = tf.layers.dense(logits_visual, 256, name='visual1')
        logits_visual = tf.layers.dense(logits_visual, 256, activation=tf.nn.relu, name='visual2')
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)

        logits_text = tf.math.l2_normalize(words, axis=1)
        logits_text = tf.layers.dense(logits_text, 256, name='text1')
        logits_text = tf.layers.dense(logits_text, 256, activation=tf.nn.relu, name='text2')
        logits_text = tf.math.l2_normalize(logits_text, axis=1)
    
    variable_map = {}
    for variable in tf.global_variables():
        var_name = variable.name.split('/')
        if var_name[0] == 'Model':
            variable_map[variable.name.replace(':0', '')] = variable
    
    predictions_normal = compute_predictions(logits_visual, logits_text, labels)
    
    probabilities = compute_probabilities(logits_visual, logits_text)
    predictions_soft = tf.argmax(probabilities, axis=1)
    top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=probabilities, targets=labels, k=3), tf.float32))
    top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=probabilities, targets=labels, k=5), tf.float32))

    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions_normal),
            'top1': tf.metrics.accuracy(labels=labels, predictions=predictions_soft),
            'top3': tf.metrics.mean(top3),
            'top5': tf.metrics.mean(top5)
        }
    
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # ================================== Testing the model =====================================

    logging.info("Testing the model")
    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    checkpoint = tf.train.latest_checkpoint(os.path.join(model_dir, 'best_weights'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint)

        sess.run(test_inputs['iterator_init_op'])
        sess.run(metrics_init_op)
        
        for _ in trange(test_inputs['num_steps']):
            sess.run(update_metrics_op)
        
        metrics_values = {k: v[0] for k, v in metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Test metrics: " + metrics_string)