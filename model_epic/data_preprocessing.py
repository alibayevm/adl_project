import numpy as np
import os

def preprocess(rgb, flow, words, split):
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
    
    np.savez(os.path.join('data', 'preprocessed.npz'), visuals=visuals, words=words, labels=labels)