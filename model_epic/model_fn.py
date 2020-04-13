import tensorflow as tf
import numpy as np 
import os
from model_epic.triplet_loss import get_avg_triplet_loss, get_valid_triplets, get_random_triplet_loss
from model_epic.triplet_loss import batch_hard_triplet_loss
from model_epic.accuracy import compute_predictions


def model_fn(inputs, params, mode):
    visuals = inputs['visuals']
    labels = inputs['labels']
    words = inputs['words']
    reuse = mode == 'valid'

    # Define the entire model
    with tf.variable_scope('Model', reuse=reuse):
        logits_visual = tf.math.l2_normalize(visuals, axis=1)
        logits_visual = tf.layers.dense(logits_visual, params.fc1, use_bias=False, name='visual1')
        logits_visual = tf.layers.dense(logits_visual, params.fc2, activation=tf.nn.relu, use_bias=False, name='visual2')
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)

        logits_text = tf.math.l2_normalize(words, axis=1)
        logits_text = tf.layers.dense(logits_text, params.fc1, use_bias=False, name='text1')
        logits_text = tf.layers.dense(logits_text, params.fc2, activation=tf.nn.relu, use_bias=False, name='text2')
        logits_text = tf.math.l2_normalize(logits_text, axis=1)


    # TODO: Change this to randomly sample `params.triplets` triplets per query without losing gradients
    """
    loss_vv = get_avg_triplet_loss(labels, logits_visual, logits_visual, params.margin, cross_modal=False)
    loss_tt = get_avg_triplet_loss(labels, logits_text, logits_text, params.margin, cross_modal=False)
    loss_vt = get_avg_triplet_loss(labels, logits_visual, logits_text, params.margin, cross_modal=True)
    loss_tv = get_avg_triplet_loss(labels, logits_text, logits_visual, params.margin, cross_modal=True)
    """
    
    """
    loss_vv = get_valid_triplets(labels, logits_visual, logits_visual, params.margin, cross_modal=False)
    loss_tt = get_valid_triplets(labels, logits_text, logits_text, params.margin, cross_modal=False)
    loss_vt = get_valid_triplets(labels, logits_visual, logits_text, params.margin, cross_modal=True)
    loss_tv = get_valid_triplets(labels, logits_text, logits_visual, params.margin, cross_modal=True)

    loss_vv = tf.reduce_sum(loss_vv)
    loss_tt = tf.reduce_sum(loss_tt)
    loss_vt = tf.reduce_sum(loss_vt)
    loss_tv = tf.reduce_sum(loss_tv)
    """

    """
    loss_vv = get_random_triplet_loss(labels, logits_visual, logits_visual, params.margin, params.num_triplets, cross_modal=False)
    loss_tt = get_random_triplet_loss(labels, logits_text, logits_text, params.margin, params.num_triplets, cross_modal=False)
    loss_vt = get_random_triplet_loss(labels, logits_visual, logits_text, params.margin, params.num_triplets, cross_modal=True)
    loss_tv = get_random_triplet_loss(labels, logits_text, logits_visual, params.margin, params.num_triplets, cross_modal=True)
    """

    loss_vv = batch_hard_triplet_loss(labels, logits_visual, logits_visual, params.margin, cross_modal=False)
    loss_tt = batch_hard_triplet_loss(labels, logits_text, logits_text, params.margin, cross_modal=False)
    loss_vt = batch_hard_triplet_loss(labels, logits_visual, logits_text, params.margin, cross_modal=True)
    loss_tv = batch_hard_triplet_loss(labels, logits_text, logits_visual, params.margin, cross_modal=True)

    total_loss = params.lambda_within * (loss_tt + loss_vv) + params.lambda_cross * (loss_vt + loss_tv)


    # Create variable maps for RGB and Flow I3D models
    # Store the top layers into a list
    if mode == 'train':
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(total_loss)
    elif mode == 'test':
        variable_map = {}
        for variable in tf.global_variables():
            var_name = variable.name.split('/')
            if var_name[0] == 'Model':
                variable_map[variable.name.replace(':0', '')] = variable
        # TODO: implement accuracy metric calculations here
        predictions = compute_predictions(logits_visual, logits_text, labels)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        if mode == 'test':
            metrics = {
                'accuracy': tf.metrics.accuracy(labels, predictions)
            }
        else:
            metrics = {
                'total_loss': tf.metrics.mean(total_loss),
                'loss_vv' : tf.metrics.mean(loss_vv),
                'loss_tt' : tf.metrics.mean(loss_tt),
                'loss_vt' : tf.metrics.mean(loss_vt),
                'loss_tv' : tf.metrics.mean(loss_tv)
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
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op

    if mode == 'train':
        model_spec['train_op'] = train_op
    elif mode == 'test':
        model_spec['variable_map'] = variable_map
    
    return model_spec

