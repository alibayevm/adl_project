from wikipedia2vec import Wikipedia2Vec
import gensim
import numpy as np
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
args = parser.parse_args()

def compute_embedding(model_name, word):
    if 'wiki' in model_name:
        model = Wikipedia2Vec.load(model_name)
        return model.get_word_vector(word)
    elif 'google' in model_name:
        model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)
        return model[word]

if __name__ == "__main__":
    model_name = args.model_name
    dimensions = int(model_name.split('.')[0][-3:])
    
    # Class keys
    manyshots = pd.read_csv('EPIC_many_shot_verbs.csv')
    class_key_vectors = []
    for _, row in manyshots.iterrows():
        words = row['verb'].strip().split('-')
        vector = np.zeros(dimensions, dtype=np.float32)
        for word in words:
            vector += compute_embedding(model_name, word)
        vector = vector / len(words)
        class_key_vectors.append(vector)
    
    # Save file
    class_key_vectors = np.stack(class_key_vectors)
    filename = '{}_classkeys.npy'.format(model_name.split('.')[0])
    np.save(os.path.join('..', 'word_embeddings', filename), class_key_vectors, allow_pickle=False, fix_imports=False)

    # Training videos
    train = open(os.path.join('..', 'splits', 'train_embedding.txt'), 'r')
    train_vecotrs = []
    
    for line in train:
        verb = line.strip().split(' ')[-1]
        words = verb.split('-')
        vector = np.zeros(dimensions, dtype=np.float32)
        for word in words:
            vector += compute_embedding(model_name, word)
        vector = vector / len(words)
        train_vecotrs.append(vector)
    
    train_vecotrs = np.stack(train_vecotrs)
    filename = '{}_train.npy'.format(model_name.split('.')[0])
    np.save(os.path.join('..', 'word_embeddings', filename), train_vecotrs, allow_pickle=False, fix_imports=False)
