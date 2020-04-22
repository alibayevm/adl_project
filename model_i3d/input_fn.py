import tensorflow as tf
import numpy as np 
import os
from PIL import Image
from model_i3d.data_augment import transform_data
import random

def load_image(clip_path, index, modality):
    """
    Loads the frame `index` at `clip_path` location.
    RGB frame is returned as a list of a single element.
    Flow frames are returned as a list with 2 elements.
    """
    filename = 'frame_{:010d}.jpg'.format(index)
    if modality == 'rgb':
        img = Image.open(os.path.join(clip_path.format('rgb', ''), filename)).convert('RGB')
        return [img]
    else:
        u_img = Image.open(os.path.join(clip_path.format('flow', 'u/'), filename)).convert('L')
        v_img = Image.open(os.path.join(clip_path.format('flow', 'v/'), filename)).convert('L')
        return [u_img, v_img]


def get_frames_and_labels(data, num_frames, sampling_rate, modality, word_embedding, is_training):
    """
    The function returns frames as a Numpy array, a class label, and embedded word vector.
    """
    # Decode strings to utf-8
    clip_path = data[0].decode("utf-8")
    total_frames = int(data[1].decode("utf-8"))
    label = np.int64(int(data[2].decode("utf-8")))
    sample_index = int(data[3].decode("utf-8"))
    modality = modality.decode("utf-8")
    word_embedding = word_embedding.decode("utf-8")

    # List that will store the loaded frames
    frames = []

    # EPIC Flow frames were sampled at 30 fps, so they have twice less frames
    if modality != 'rgb':
        total_frames = total_frames // 2
        sampling_rate = (sampling_rate + 1) // 2

    # Choose the first frame where to start sampling
    if is_training:
        interval_size = sampling_rate * (num_frames - 1) + 1
        start = random.randint(0, max(0, total_frames - interval_size))
        stop = start + num_frames * sampling_rate
    else:
        start = 0
        stop = max(total_frames, sampling_rate * num_frames)
    
    # Get list of frames
    # In case if sampling_rate is zero, we will sample uniformly
    if sampling_rate > 0:
        for i in range(start, stop, sampling_rate):
            index = (i % total_frames) + 1
            frames.extend(load_image(clip_path, index, modality))
    else:
        sampling_rate = total_frames // num_frames
        for i in range(num_frames):
            start = int(i * sampling_rate)
            end = max(start, int((i+1) * sampling_rate) - 1)
            index = random.randint(start, end) + 1
            frames.extend(load_image(clip_path, index, modality))
            
    # Apply transofrmations
    frames = transform_data(frames, random_crop=is_training, random_flip=is_training)

    # Convert frames to numpy arrays
    frames_np = []
    if modality == 'rgb':
        for frame in frames:
            frames_np.append(np.asarray(frame))
    else:
        for i in range(0, len(frames), 2):
            frame = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
            frames_np.append(frame)
    frames = np.stack(frames_np)

    # Normalize frames
    frames = frames / 255 * 2 - 1
    frames = np.array(frames, dtype='float32')

    # Get word embedding
    if is_training:
        words = np.load(os.path.join('data', 'word_embeddings', 'train.npy'))
        word = words[sample_index]
    else:
        word = np.zeros(int(word_embedding[-3:]))
        
    return frames, word.astype(np.float32), label


def parse_fn(data, params, is_training):
    """
    A wrapper function to call `get_frames_and_labels` function
    """
    [frames, word, label] = tf.py_func(
        get_frames_and_labels,
        [data, params.num_frames, params.sampling_rate, "rgb", params.word_embedding, is_training],
        [tf.float32, tf.float32, tf.int64]
    )

    return frames, word, label


def input_fn(data_list, params, is_training):
    """
    Takes the dataset information and returns processed dataset
    """
    # Functions that will convert video path into video frames, labels, and word embeddings
    parse_train = lambda d: parse_fn(d, params, True)
    parse_test = lambda d: parse_fn(d, params, False)

    # Make a Dataset object that will read the list of data
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_list)))
            .shuffle(len(data_list))
            .map(parse_train, num_parallel_calls=params.num_parallel_calls)
            .repeat()
            .batch(params.batch_size)
            .prefetch(1)
        )
        num_steps = (len(data_list) - 1) // params.batch_size + 1
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(data_list)))
            .map(parse_test, num_parallel_calls=params.num_parallel_calls)
            .repeat()
            .batch(params.batch_size)
            .prefetch(1)
        )
        num_steps = (len(data_list) - 1) // params.batch_size + 1
        #num_steps = len(data_list)
    
    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    clips, words, labels = iterator.get_next()

    if not is_training:
        words = np.load(os.path.join('data', 'word_embeddings', 'class_keys.npy'))
        words = tf.constant(words, dtype=tf.float32)
    
    clips.set_shape([None, None, 224, 224, 3])
    words.set_shape([None, 100])
    iterator_init_op = iterator.initializer

    inputs = {
        'clips': clips, 
        'labels': labels, 
        'words': words, 
        'iterator_init_op': iterator_init_op, 
        'num_steps' : num_steps
    }

    return inputs
