import tensorflow as tf
from model_i3d.i3d import InceptionI3d
import numpy as np 
import os
from model_i3d.triplet_loss import get_valid_triplets

def sum_random_triplets(triplets, num_samples):
    total_loss = 0.0
    for query in triplets:
        flat = query.flatten()
        flat = flat[np.nonzero(flat)]
        flat = np.random.choice(flat, num_samples, replace=False)
        total_loss += np.sum(flat)
    return np.float(total_loss)


def model_fn(inputs, params, is_training):
    clips_rgb = inputs['clips_rgb']
    clips_flow = inputs['clips_flow']
    labels = inputs['labels']
    words = inputs['words']
    reuse = not is_training

    # Define the entire model
    with tf.variable_scope('Model', reuse=reuse):
        # Define the RGB model
        with tf.variable_scope('RGB', reuse=reuse):
            model_rgb = InceptionI3d(num_classes=params.feature_size)
            logits_rgb, _ = model_rgb(clips_rgb, is_training=is_training, dropout_keep_prob=1.0)

        # Define the Flow model
        with tf.variable_scope('Flow', reuse=reuse):
            model_flow = InceptionI3d(num_classes=params.feature_size)
            logits_flow, _ = model_flow(clips_flow, is_training=is_training, dropout_keep_prob=1.0)

        # Embedding step
        logits_visual = tf.concat([logits_rgb, logits_flow], 1)
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)
        logits_visual = tf.layers.dense(logits_visual, params.fc1, use_bias=False, name='visual1')
        logits_visual = tf.layers.dense(logits_visual, params.fc2, activation=tf.nn.relu, use_bias=False, name='visual2')
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)

        logits_text = tf.math.l2_normalize(words, axis=1)
        logits_text = tf.layers.dense(logits_text, params.fc1, use_bias=False, name='text1')
        logits_text = tf.layers.dense(logits_text, params.fc2, activation=tf.nn.relu, use_bias=False, name='text2')
        logits_text = tf.math.l2_normalize(logits_text, axis=1)

        # Names of the variables that will be tuned for first epochs
        top_variables = ['Logits', 'visual1', 'visual2', 'text1', 'text2']


    # valid triplet losses
    loss_vv = get_valid_triplets(labels, logits_visual, logits_visual, params.margin, cross_modal=False)
    loss_tt = get_valid_triplets(labels, logits_text, logits_text, params.margin, cross_modal=False)
    loss_vt = get_valid_triplets(labels, logits_visual, logits_text, params.margin, cross_modal=True)
    loss_tv = get_valid_triplets(labels, logits_text, logits_visual, params.margin, cross_modal=True)

    # HACK: Changed this to randomly sample `params.triplets` triplets per query
    # loss_vv = tf.reduce_sum(loss_vv)
    # loss_tt = tf.reduce_sum(loss_tt)
    # loss_vt = tf.reduce_sum(loss_vt)
    # loss_tv = tf.reduce_sum(loss_tv)
    [loss_vv] = tf.py_func(sum_random_triplets, [loss_vv, params.num_triplets], [tf.float32])
    [loss_tt] = tf.py_func(sum_random_triplets, [loss_tt, params.num_triplets], [tf.float32])
    [loss_vt] = tf.py_func(sum_random_triplets, [loss_vt, params.num_triplets], [tf.float32])
    [loss_tv] = tf.py_func(sum_random_triplets, [loss_tv, params.num_triplets], [tf.float32])

    total_loss = params.lambda_within * (loss_tt + loss_vv) + params.lambda_cross * (loss_vt + loss_tv)


    # Create variable maps for RGB and Flow I3D models
    # Store the top layers into a list
    if is_training:
        variable_map_rgb = {}
        variable_map_flow = {}
        train_vars = []

        for variable in tf.global_variables():
            var_name = variable.name.split('/')
            if var_name[0] != 'Model':
                continue
            if var_name[1] == 'RGB' and var_name[2] == 'inception_i3d' and var_name[3] not in top_variables:
                variable_map_rgb[variable.name.replace(':0', '')] = variable
            if var_name[1] == 'Flow' and var_name[2] == 'inception_i3d' and var_name[3] not in top_variables:
                variable_map_flow[variable.name.replace(':0', '')] = variable
            if var_name[3] in top_variables or var_name[1] in top_variables:
                train_vars.append(variable)

        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op_initial = optimizer.minimize(total_loss, var_list=train_vars)
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
        model_spec['train_op_initial'] = train_op_initial
        model_spec['train_op'] = train_op
        model_spec['variable_map_rgb'] = variable_map_rgb   
        model_spec['variable_map_flow'] = variable_map_flow
    else:
        model_spec['variable_map'] = variable_map
    
    return model_spec

