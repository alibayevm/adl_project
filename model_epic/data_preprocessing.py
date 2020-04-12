import numpy as np
import os

def preprocess(rgb, flow, words, split, is_training):
    rgb = np.load(rgb)
    flow = np.load(flow)
    words = np.load(words)
    split = open(split)

    visuals = np.stack([rgb, flow], axis=1)
    visuals = np.average(visuals, axis=(1,2))

    labels = []
    for line in split:
        label = int(line.strip().split(' ')[2])
        labels.append(label)
    labels = np.stack(labels)

    filename = 'preprocessed_train.npz' if is_training else 'preprocessed_valid.npz'
    
    np.savez(os.path.join('data', filename), visuals=visuals, words=words, labels=labels)

def preprocess_test(rgb, flow, words, split):
    rgb = np.load(rgb)
    flow = np.load(flow)
    class_words = np.load(words)
    split = open(split)

    visuals = np.stack([rgb, flow], axis=1)
    visuals = np.average(visuals, axis=(1,2))

    labels = []
    words = []

    for line in split:
        label = int(line.strip().split(' ')[2])
        labels.append(label)
        words.append(class_words[label])
    
    labels = np.stack(labels)
    words = np.stack(words)

    np.savez(os.path.join('data', 'preprocessed_test.npz'), visuals=visuals, words=words, labels=labels)