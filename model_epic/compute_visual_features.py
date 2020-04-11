import torch.hub
import os
import numpy as np
from PIL import Image
import random
from data_augment import transform_data
import argparse
from tqdm import tqdm

repo = 'epic-kitchens/action-models'

class_counts = (125, 352)
segment_count = 8
snippet_length = {
    'rgb' : 1,
    'flow' : 5
}
height, width = 224, 224


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


def get_frames(clip_path, total_frames, modality, is_training=True, num_segments=8):
    if modality != 'rgb':
        total_frames = total_frames // 2
    sampling_rate = max(total_frames // num_segments, snippet_length[modality])

    frames = []

    for i in range(num_segments):
        start = i * sampling_rate
        end = (i+1) * sampling_rate - 1
        index = random.randint(start, end)
        index = min(index, end+1 - snippet_length[modality])

        for j in range(snippet_length[modality]):
            frame_index = (index + j) % total_frames + 1
            frames.extend(load_image(clip_path, frame_index, modality))

    frames = transform_data(frames, random_crop=is_training, random_flip=is_training)
    frames_np = []

    if modality != 'rgb':
        for i in range(0, len(frames), 2):
            frame_np = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
            frame_np = np.transpose(frame_np, (2, 0, 1))
            frames_np.append(frame_np)
    else:
        for frame in frames:
            frame_np = np.asarray(frame)
            frame_np = np.transpose(frame_np, (2, 0, 1))
            frames_np.append(frame_np)

    segments = []
    for i in range(0, len(frames_np), snippet_length[modality]):
        snippet = np.stack(frames_np[i : i+snippet_length[modality]])
        segments.append(snippet)
    
    segments = np.stack(segments)
    segments = segments / 255 * 2 - 1
    
    # Batch of size 1
    segments = np.stack([segments])

    return torch.from_numpy(segments.astype(np.float32))
    

parser = argparse.ArgumentParser()
parser.add_argument('modality', help='Frame modality')
args = parser.parse_args()


if __name__ == "__main__":

    base_model = 'resnet50'
    
    modality = args.modality

    if modality == 'rgb':
        tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                            base_model=base_model, 
                            pretrained='epic-kitchens', force_reload=True)
    elif modality == 'flow':
        tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'Flow',
                            base_model=base_model, 
                            pretrained='epic-kitchens', force_reload=True)

    # Training features
    train_embedding = open(os.path.join('..', 'data', 'splits', 'train_embedding.txt'))
    all_features = []

    for line in tqdm(train_embedding):
        clip_path, total_frames = line.strip().split(' ')[:2]
        inputs = get_frames(clip_path, int(total_frames), modality)
        inputs = inputs.reshape((1, -1, height, width))

        features = tsn.features(inputs)
        features = features.detach().numpy()
        all_features.append(features)

    all_features = np.stack(all_features)
    np.save(os.path.join('visual_features', 'tsn_resnet50_{}.npy'.format(modality)), all_features, allow_pickle=False, fix_imports=False)


        

