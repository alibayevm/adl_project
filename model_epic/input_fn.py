import tensorflow as tf
import numpy as np 
from tqdm import trange

def input_fn(visuals, words, labels, params, is_training):
    """
    Takes the dataset information and returns processed dataset
    """
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((visuals, words, labels))
            .shuffle(len(labels))
            .repeat()
            .batch(params.batch_size)
            .prefetch(params.prefetch_train)
        )
        num_steps = (len(labels) - 1) // params.batch_size + 1
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((visuals, words, labels))
            .repeat()
            .batch(1)
            .prefetch(params.prefetch_test)
        )
        num_steps = len(labels)
    
    # Dimensionality of word vector
    word_dims = int(params.word_embedding[-3:])

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    visuals, words, labels = iterator.get_next()
    visuals.set_shape([None, 2048])
    words.set_shape([None, word_dims])
    iterator_init_op = iterator.initializer

    inputs = {
        'visuals': visuals, 
        'words': words,
        'labels': labels, 
        'iterator_init_op': iterator_init_op, 
        'num_steps' : num_steps
    }

    return inputs

def check_dataset(inputs):
    with tf.Session() as sess:
        sess.run(inputs['iterator_init_op'])
        t = trange(inputs['num_steps'])

        for _ in t:
            visuals, words, labels = sess.run([inputs['visuals'], inputs['words'], inputs['labels']])
        print(visuals.shape)
        print(words.shape)
        print(labels.shape)