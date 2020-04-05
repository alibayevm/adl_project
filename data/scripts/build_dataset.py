import pandas as pd
import os
import random

_dataset_path = '/home/maxat/action_recognition/starter-kit-action-recognition/data/interim/{:s}_train_segments'

def save_to_txt(dataset_indices, df, filename):
    file = open(os.path.join('..', 'splits', filename), 'w')
    for verb_class, indices in dataset_indices.items():
        for index in indices:
            video = df.loc[index]
            participant_id = video['participant_id']
            video_id = video['video_id']
            narration = '-'.join(video['narration'].strip().split(' '))
            
            video_path = '{}/{}/{}/{}{}_{}_{}'.format(_dataset_path, participant_id, video_id, '{:s}', video_id, index, narration)
            num_frames = int(video['stop_frame']) - int(video['start_frame'])
            
            file.write('{} {} {} {}\n'.format(video_path, num_frames, verb_class, video['verb']))
    
    file.close()


def split_dataset(train_frac=60, valid_frac=20):
    """
    Splits the entire EPIC train dataset into train, validation, and test sets. 
    Keeps class distributions identical for all sets.
    Validation set is sampled from training set. 
    """
    # Get many shot classes and map them to new class labels
    df_manyshot = pd.read_csv('EPIC_many_shot_verbs.csv')
    manyshot_newclass = {}
    manyshot_indices = {}
    manyshot_verbs = []
    for _, row in df_manyshot.iterrows():
        manyshot_newclass[int(row['verb_class'])] = int(row['new_class'])
        manyshot_indices[int(row['new_class'])] = []
        manyshot_verbs.append(int(row['verb_class']))

    # EPIC dataset
    df = pd.read_csv('EPIC_train_action_labels.csv')

    # Get list of indicies for each class in the dataframe
    for i, row in df.iterrows():
        if int(row['verb_class']) in manyshot_verbs:
            new_class = manyshot_newclass[int(row['verb_class'])]
            manyshot_indices[new_class].append(i)
    
    # Split train and test sets
    test_frac = 1.0 - (train_frac + valid_frac) / 100.0
    test_indices = {}
    train_indices_embedding = {}

    for verb_class, indices in manyshot_indices.items():
        test_indices[verb_class] = random.sample(indices, int(len(indices) * test_frac))
        train_indices_embedding[verb_class] = []
        for i in indices:
            if i not in test_indices[verb_class]:
                train_indices_embedding[verb_class].append(i)
    
        
    # Split train and validation sets
    valid_frac = valid_frac / (train_frac + valid_frac)
    train_indices_onehot = {}
    valid_indices = {}

    for verb_class, indices in train_indices_embedding.items():
        valid_indices[verb_class] = random.sample(indices, int(len(indices) * valid_frac))
        train_indices_onehot[verb_class] = []
        for i in indices:
            if i not in valid_indices[verb_class]:
                train_indices_onehot[verb_class].append(i)

    # Write the dataset information to the text files
    save_to_txt(train_indices_embedding, df, 'train_embedding.txt')
    save_to_txt(train_indices_onehot, df, 'train_onehot.txt')
    save_to_txt(valid_indices, df, 'valid.txt')
    save_to_txt(test_indices, df, 'test.txt')


if __name__ == "__main__":
    split_dataset()