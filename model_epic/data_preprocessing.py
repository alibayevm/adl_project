import numpy as np
import os

def preprocess(rgb, flow, words, split, is_training, onehot=False):
    rgb = np.load(rgb)
    flow = np.load(flow)
    words = np.load(words)
    split = open(split)

    visuals = np.stack([rgb, flow], axis=1)
    visuals = np.average(visuals, axis=(1,2))

    rgb = np.average(rgb, axis=1)
    flow = np.average(flow, axis=1)

    labels = []
    for line in split:
        label = int(line.strip().split(' ')[2])
        labels.append(label)
    labels = np.stack(labels)

    postfix = '_onehot' if onehot else ''
    filename = 'preprocessed_train{}.npz'.format(postfix) if is_training else 'preprocessed_valid{}.npz'.format(postfix)
    
    if onehot:
        np.savez(os.path.join('data', filename), rgb=rgb, flow=flow, labels=labels)
    else:
        np.savez(os.path.join('data', filename), visuals=visuals, words=words, labels=labels)

def preprocess_test(rgb, flow, split, onehot=False):
    rgb = np.load(rgb)
    flow = np.load(flow)
    split = open(split)

    visuals = np.stack([rgb, flow], axis=1)
    visuals = np.average(visuals, axis=(1,2))

    rgb = np.average(rgb, axis=1)
    flow = np.average(flow, axis=1)

    labels = []

    for line in split:
        label = int(line.strip().split(' ')[2])
        labels.append(label)
    
    labels = np.stack(labels)

    if onehot:
        np.savez(os.path.join('data', 'preprocessed_test_onehot.npz'), rgb=rgb, flow=flow, labels=labels)
    else:
        np.savez(os.path.join('data', 'preprocessed_test.npz'), visuals=visuals, labels=labels)