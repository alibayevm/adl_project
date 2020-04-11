import tensorflow as tf
import numpy as np 
import os
from model_epic.triplet_loss import get_avg_triplet_loss

def model_fn(inputs, params, is_training):
    visuals = inputs['visuals']
    labels = inputs['labels']
    words = inputs['words']
    reuse = not is_training

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
    loss_vv = get_avg_triplet_loss(labels, logits_visual, logits_visual, params.margin, cross_modal=False)
    loss_tt = get_avg_triplet_loss(labels, logits_text, logits_text, params.margin, cross_modal=False)
    loss_vt = get_avg_triplet_loss(labels, logits_visual, logits_text, params.margin, cross_modal=True)
    loss_tv = get_avg_triplet_loss(labels, logits_text, logits_visual, params.margin, cross_modal=True)

    total_loss = params.lambda_within * (loss_tt + loss_vv) + params.lambda_cross * (loss_vt + loss_tv)


    # Create variable maps for RGB and Flow I3D models
    # Store the top layers into a list
    if is_training:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(total_loss)
    else:
        variable_map = {}
        for variable in tf.global_variables():
            var_name = variable.name.split('/')
            if var_name[0] == 'Model':
                variable_map[variable.name.replace(':0', '')] = variable

    # TODO: add metrics tensor and ops to update it

    # MODEL SPECIFICATION
    model_spec = inputs
    model_spec['loss_vv'] = loss_vv
    model_spec['loss_tt'] = loss_tt
    model_spec['loss_vt'] = loss_vt
    model_spec['loss_tv'] = loss_tv
    model_spec['total_loss'] = total_loss

    if is_training:
        model_spec['train_op'] = train_op
    else:
        model_spec['variable_map'] = variable_map
    
    return model_spec

