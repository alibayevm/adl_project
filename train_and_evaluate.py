import tensorflow as tf
import numpy as np
import os
from model_epic.triplet_loss import get_avg_triplet_loss, get_random_triplet_loss
from model_epic.accuracy import compute_predictions
from tqdm import trange

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Hyper Parameters
batch_size = 128
margin = 1.0
learning_rate = 1e-5
dropout_val = 0.6
num_epochs = 300
embedding = True
random = True


# Embedding model
def model_fn_emb(model, is_training):
    dropout = dropout_val if is_training else 1.0
    labels = tf.cast(model['labels'], tf.int64)
    
    train_op = None
    conf_mat = None
    reuse = not is_training

    with tf.variable_scope('Model', reuse=reuse):
        logits_visual = tf.math.l2_normalize(model['visuals'], axis=1)
        logits_visual = tf.layers.dense(logits_visual, 256, name='visual1')
        logits_visual = tf.layers.dropout(logits_visual, rate=dropout)
        logits_visual = tf.layers.dense(logits_visual, 256, activation=tf.nn.relu, name='visual2')
        logits_visual = tf.math.l2_normalize(logits_visual, axis=1)

        logits_text = tf.math.l2_normalize(model['words'], axis=1)
        logits_text = tf.layers.dense(logits_text, 256, name='text1')
        logits_text = tf.layers.dropout(logits_text, rate=dropout)
        logits_text = tf.layers.dense(logits_text, 256, activation=tf.nn.relu, name='text2')
        logits_text = tf.math.l2_normalize(logits_text, axis=1)
        
    if is_training:
        if random:
            vv = get_random_triplet_loss(labels, logits_visual, logits_visual, margin, 100, cross_modal=False)
            tt = get_random_triplet_loss(labels, logits_text, logits_text, margin, 100, cross_modal=False)
            vt = get_random_triplet_loss(labels, logits_visual, logits_text, margin, 100, cross_modal=True)
            tv = get_random_triplet_loss(labels, logits_text, logits_visual, margin, 100, cross_modal=True)
        else:
            vv = get_avg_triplet_loss(labels, logits_visual, logits_visual, margin, cross_modal=False)
            tt = get_avg_triplet_loss(labels, logits_text, logits_text, margin, cross_modal=False)
            vt = get_avg_triplet_loss(labels, logits_visual, logits_text, margin, cross_modal=True)
            tv = get_avg_triplet_loss(labels, logits_text, logits_visual, margin, cross_modal=True)
    
        total = 0.1 * (vv + tt) + 1.0 * (vt + tv)

        with tf.variable_scope("metrics"):
            metrics = {
                'loss': tf.metrics.mean(total)
            }

        optim = tf.train.AdamOptimizer(learning_rate)
        train_op = optim.minimize(total)
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

    model['train_op'] = train_op
    model['metrics_init_op'] = metrics_init_op
    model['metrics'] = metrics
    model['update_metrics_op'] = update_metrics_op
    model['conf_mat'] = conf_mat
    
    return model


# One hot encoding model
def model_fn_onehot(model, is_training):
    labels = tf.cast(model['labels'], tf.int64)
    train_op = None
    reuse = not is_training
    dropout = dropout_val if is_training else 1.0

    with tf.variable_scope('Model', reuse=reuse):
        logits_visual = tf.layers.dense(model['visuals'], 256, name='visual1')
        logits_visual = tf.layers.dropout(logits_visual, rate=dropout)
        logits_visual = tf.nn.relu(logits_visual)
        logits_visual = tf.layers.dense(logits_visual, 26)
    
    predictions = tf.argmax(logits_visual, 1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_visual)
    conf_mat = tf.confusion_matrix(labels, predictions, num_classes=26)

    if is_training:
        optim = tf.train.AdamOptimizer(learning_rate)
        train_op = optim.minimize(loss)
        
    with tf.variable_scope("metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)
        }
            
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    model['train_op'] = train_op
    model['metrics_init_op'] = metrics_init_op
    model['metrics'] = metrics
    model['update_metrics_op'] = update_metrics_op
    model['conf_mat'] = conf_mat
    
    return model
    

def list_of_datasets(visuals, words, labels, num_classes=26):
    visual_dim = visuals.shape[1]
    labels = np.reshape(labels, (len(labels), 1))
    data = np.concatenate([visuals, words, labels], axis=1)
    
    classes = [[] for _ in range(num_classes)]
	
    for example in data:
        classes[int(example[-1])].append(example)

    datasets = [None for _ in range(num_classes)]

    for i, class_list in enumerate(classes):
        class_data = np.stack(class_list)
        v, w, l = class_data[:,:visual_dim], class_data[:,visual_dim:-1], class_data[:,-1]
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(v, dtype=tf.float32), tf.constant(w, dtype=tf.float32), tf.constant(l, dtype=tf.float32)))
        dataset = dataset.shuffle(len(l)).repeat()
        datasets[i] = dataset

    return datasets



if __name__ == "__main__":
    tf.set_random_seed(2020)

    #=========================================== Load data from .npy files =====================================================

    # ========================================
    # Train data
    train_rgb = np.load(os.path.join('model_epic', 'visual_features', 'train', 'tsn_resnet50_rgb.npy'))
    train_flow = np.load(os.path.join('model_epic', 'visual_features', 'train', 'tsn_resnet50_flow.npy'))
    train_labels = []
    file = open(os.path.join('data', 'splits', 'train.txt'))
    for line in file:
        train_labels.append(int(line.strip().split(' ')[2]))

    train_rgb = np.average(train_rgb, axis=1)
    train_flow = np.average(train_flow, axis=1)
    # NOTE: Concatenated RGB and Flow instead of averaging them
    train_visual = np.concatenate([train_rgb, train_flow], axis=1)
    train_words = np.load(os.path.join('data', 'word_embeddings', 'w2v_wiki_100_train.npy'))
    train_labels = np.stack(train_labels)

    # ========================================
    # Validation data
    valid_rgb = np.load(os.path.join('model_epic', 'visual_features', 'valid', 'tsn_resnet50_rgb.npy'))
    valid_flow = np.load(os.path.join('model_epic', 'visual_features', 'valid', 'tsn_resnet50_flow.npy'))
    valid_labels = []
    file = open(os.path.join('data', 'splits', 'valid.txt'))
    for line in file:
        valid_labels.append(int(line.strip().split(' ')[2]))

    valid_rgb = np.average(valid_rgb, axis=1)
    valid_flow = np.average(valid_flow, axis=1)
    valid_visual = np.concatenate([valid_rgb, valid_flow], axis=1)
    valid_words = np.load(os.path.join('data', 'word_embeddings', 'w2v_wiki_100_valid.npy'))
    valid_labels = np.stack(valid_labels)


    #========================================= Define dataset objects and models ==================================================

    # ========================================
    # Training dataset and model
    # TODO: Balance the dataset
    
    # Class distributions
    class_distr = np.zeros(26)
    for label in train_labels:
        class_distr[label] += 1
    class_distr /= np.sum(class_distr)
    balanced_class_distr = np.clip(class_distr, 0.0, 0.1)
    balanced_class_distr = np.exp(balanced_class_distr) / sum(np.exp(balanced_class_distr))

    train_datasets = list_of_datasets(train_visual, train_words, train_labels)
    train_dataset = tf.data.experimental.sample_from_datasets(
                        train_datasets, balanced_class_distr).batch(batch_size)


    # train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_visual), tf.constant(train_words), tf.constant(train_labels)))
    # train_dataset = train_dataset.shuffle(len(train_labels))
    # train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(1)

    train_iterator = train_dataset.make_initializable_iterator()
    visuals, words, labels = train_iterator.get_next()
    train_iterator_init = train_iterator.initializer
    num_steps = len(train_labels) // batch_size

    train_model = {
        'iterator_init_op': train_iterator_init,
        'visuals': visuals,
        'words': words,
        'labels': labels,
        'num_steps': num_steps
    }

    if embedding:
        train_model = model_fn_emb(train_model, True)
    else:
        train_model = model_fn_onehot(train_model, True)

    

    # ========================================
    # Validation dataset and model
    class_keys = tf.constant(np.load(os.path.join('data', 'word_embeddings', 'w2v_wiki_100_classkeys.npy')))

    valid_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(valid_visual), tf.constant(valid_words), tf.constant(valid_labels)))
    #valid_dataset = valid_dataset.shuffle(len(valid_labels))
    valid_dataset = valid_dataset.batch(batch_size)
    valid_dataset = valid_dataset.prefetch(1)

    valid_iterator = valid_dataset.make_initializable_iterator()
    visuals, words, labels = valid_iterator.get_next()
    valid_iterator_init = valid_iterator.initializer
    num_steps = len(valid_labels) // batch_size

    valid_model = {
        'visuals': visuals,
        'words': class_keys,
        'labels': labels,
        'iterator_init_op': valid_iterator_init,
        'num_steps': num_steps
    }

    if embedding:
        valid_model = model_fn_emb(valid_model, False)
    else:
        valid_model = model_fn_onehot(valid_model, False)

    


    # ========================================================= Training and evaluation ======================================================
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_acc = 0.0
        best_conf_mat = np.zeros((26, 26))

        for epoch in range(num_epochs):
            # ========================================
            # Train session
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            train_op = train_model['train_op']
            update_metrics_op = train_model['update_metrics_op']
            metrics = train_model['metrics']

            sess.run(train_model['iterator_init_op'])
            sess.run(train_model['metrics_init_op'])

            for _ in range(train_model['num_steps']):
                sess.run([train_op, update_metrics_op])
            
            metrics_values = {k: v[0] for k, v in metrics.items()}
            metrics_val = sess.run(metrics_values)
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
            print("- Train metrics: " + metrics_string)

            # ========================================
            # Evaluation
            update_metrics_op = valid_model['update_metrics_op']
            metrics = valid_model['metrics']
            conf_mat_batch_op = valid_model['conf_mat']
            conf_mat = np.zeros((26, 26), dtype=np.int64)

            sess.run(valid_model['iterator_init_op'])
            sess.run(valid_model['metrics_init_op'])

            for _ in range(valid_model['num_steps']):
                _, conf_mat_batch = sess.run([update_metrics_op, conf_mat_batch_op])
                conf_mat += conf_mat_batch


            metrics_values = {k: v[0] for k, v in metrics.items()}
            metrics_val = sess.run(metrics_values)
            metrics_string = " ; ".join("{}: {:5.3f}".format(k, v) for k, v in metrics_val.items())
            print("- Valid metrics: " + metrics_string)

            if metrics_val['accuracy'] > best_acc:
                best_acc = metrics_val['accuracy']
                best_conf_mat = conf_mat
        
        print('\n\nThe best accuracy: {:.1f}%'.format(best_acc * 100))
        print('\n\nConfusion matrix:')
        for row in best_conf_mat:
            for elem in row:
                print('{:^6}'.format(elem), end='')
            print()
        print('\n\n{} classes'.format(np.sum(best_conf_mat)))
        
        
        # NOTE: can calculate Precision/Recall/F1 from best_conf_mat numpy array
        precisions = [None for _ in range(26)]
        recalls = [None for _ in range(26)]

        print('\n\n{:^20}{:^20}{:^20}\n'.format('Class', 'Precision', 'Recall'))
        for i in range(26):
            precisions[i] = best_conf_mat[i][i] / sum(np.transpose(best_conf_mat)[i])
            recalls[i] = best_conf_mat[i][i] / sum(best_conf_mat[i])
            print('{:^20}{:^20}{:^20}'.format(i, precisions[i], recalls[i]))
        print()
        precisions = np.stack(precisions)
        recalls = np.stack(recalls)

        extension = 'embed' if embedding else 'onehot'

        # Save data
        np.savez('metrics_{}.npz'.format(extension), conf_matrix=best_conf_mat, precisions=precisions, recalls=recalls)
        
