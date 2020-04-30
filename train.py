import tensorflow as tf
import numpy as np
import os
import argparse
import logging

from model_epic.utils import Params
from model_epic.utils import set_logger
from model_epic.utils import data_info
from model_epic.data_preprocessing import preprocess
from model_epic.input_fn import input_fn, input_onehot
from model_epic.model_fn import model_fn, model_onehot
from model_epic.training import train_and_validate_model
from model_epic import training_onehot



parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Path to the directory with `params.json` file')
parser.add_argument('-p', '--preprocess', action='store_true', help='Whether to preprocess input data or not')
parser.add_argument('-o', '--onehot', action='store_true', help='Whether to train onehot classifier or not')
args = parser.parse_args()

def check_dataset(inputs):
    with tf.Session() as sess:
        sess.run(inputs['iterator_init_op'])
        for i in range(5):
            print(sess.run(inputs['clips']).shape)

if __name__ == "__main__":
    tf.set_random_seed(2020)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    model_dir = args.model_dir

    # Set parameters
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))

    # Input data pipeline
    # TODO: Make it an argument
    if not os.path.isfile(os.path.join('data', 'preprocessed_train.npz')) or args.preprocess:
        visual_rgb = os.path.join('model_epic', 'visual_features', 'train', 'tsn_resnet50_rgb.npy')
        visual_flow = os.path.join('model_epic', 'visual_features', 'train', 'tsn_resnet50_flow.npy')
        words = os.path.join('data', 'word_embeddings', '{}_train.npy'.format(params.word_embedding))
        split = os.path.join('data', 'splits', 'train.txt')
        preprocess(visual_rgb, visual_flow, words, split, is_training=True, onehot=args.onehot)
    
    if not os.path.isfile(os.path.join('data', 'preprocessed_valid.npz')) or args.preprocess:
        visual_rgb = os.path.join('model_epic', 'visual_features', 'valid', 'tsn_resnet50_rgb.npy')
        visual_flow = os.path.join('model_epic', 'visual_features', 'valid', 'tsn_resnet50_flow.npy')
        words = os.path.join('data', 'word_embeddings', '{}_valid.npy'.format(params.word_embedding))
        split = os.path.join('data', 'splits', 'valid.txt')
        preprocess(visual_rgb, visual_flow, words, split, is_training=False, onehot=args.onehot)

    logging.info("Creating the dataset...")

    train_data = data_info('train')
    train_inputs = input_fn(train_data, params, is_training=True)

    valid_data = data_info('valid')
    valid_inputs = input_fn(valid_data, params, False)
    
    #check_dataset(train_inputs)
    #check_dataset(valid_inputs)
    # Define the model
    logging.info("Building the model")
    train_model_spec = model_fn(train_inputs, params, is_training=True)
    valid_model = model_fn(valid_inputs, params, False)

    # Train the model
    # TODO: Implement for onehot
    logging.info("Training the model")
    train_model(train_model_spec, valid_model, model_dir, params)
