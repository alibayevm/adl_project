import tensorflow as tf
from model_i3d.i3d import InceptionI3d
import numpy as np 
import os
from model_i3d.triplet_loss import get_avg_triplet_loss
from model_i3d.accuracy import compute_predictions


def model_fn(inputs, params, is_training):
    clips = inputs['clips']
    labels = inputs['labels']
    words = inputs['words']
    reuse = not is_training

    # Define the entire model
    with tf.variable_scope('Model', reuse=reuse):
        # Define the RGB model
        with tf.variable_scope('RGB', reuse=reuse):
            model = InceptionI3d(num_classes=1024)
            logits_visual, _ = model(clips, is_training=is_training, dropout_keep_prob=1.0)

        # Embedding step
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)
        logits_visual = tf.layers.dense(logits_visual, 256, name='visual1')
        logits_visual = tf.layers.dense(logits_visual, 256, activation=tf.nn.relu, name='visual2')
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)

        logits_text = tf.math.l2_normalize(words, axis=1)
        logits_text = tf.layers.dense(logits_text, 256, name='text1')
        logits_text = tf.layers.dense(logits_text, 256, activation=tf.nn.relu, name='text2')
        logits_text = tf.math.l2_normalize(logits_text, axis=1)

        # Names of the variables that will be tuned for first epochs
        top_variables = ['Logits', 'visual1', 'visual2', 'text1', 'text2']


    # valid triplet losses
    loss_vv = get_avg_triplet_loss(labels, logits_visual, logits_visual, params.margin, cross_modal=False)
    loss_tt = get_avg_triplet_loss(labels, logits_text, logits_text, params.margin, cross_modal=False)
    loss_vt = get_avg_triplet_loss(labels, logits_visual, logits_text, params.margin, cross_modal=True)
    loss_tv = get_avg_triplet_loss(labels, logits_text, logits_visual, params.margin, cross_modal=True)

    total_loss = params.lambda_within * (loss_tt + loss_vv) + params.lambda_cross * (loss_vt + loss_tv)


    # Create variable maps for RGB and Flow I3D models
    # Store the top layers into a list
    if is_training:
        variable_map = {}
        train_vars = []

        for variable in tf.global_variables():
            var_name = variable.name.split('/')
            if var_name[0] != 'Model':
                continue
            if var_name[1] == 'RGB' and var_name[2] == 'inception_i3d' and var_name[3] not in top_variables:
                variable_map[variable.name.replace(':0', '')] = variable
            if var_name[3] in top_variables or var_name[1] in top_variables:
                train_vars.append(variable)

        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op_initial = optimizer.minimize(total_loss, var_list=train_vars)
            train_op = optimizer.minimize(total_loss)
        with tf.variable_scope("metrics"):
            metrics = {
                'loss': tf.metrics.mean(total_loss)
            }
    else:
        predictions = compute_predictions(logits_visual, logits_text, labels)
        conf_mat = tf.confusion_matrix(labels, predictions, num_classes=26)
        with tf.variable_scope("metrics"):
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            }
    
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables) 

    # MODEL SPECIFICATION
    model_spec = inputs
    model_spec['loss_vv'] = loss_vv
    model_spec['loss_tt'] = loss_tt
    model_spec['loss_vt'] = loss_vt
    model_spec['loss_tv'] = loss_tv
    model_spec['total_loss'] = total_loss
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['update_metrics_op'] = update_metrics_op
    model_spec['metrics'] = metrics

    if is_training:
        model_spec['train_op_initial'] = train_op_initial
        model_spec['train_op'] = train_op
        model_spec['variable_map'] = variable_map 
    else:
        model_spec['conf_mat'] = conf_mat
    
    return model_spec

