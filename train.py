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

if __name__ == "__main__":
    tf.set_random_seed(230)
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

    postfix = '_onehot' if args.onehot else ''
    train_data = np.load(os.path.join('data', 'preprocessed_train{}.npz'.format(postfix)))
    valid_data = np.load(os.path.join('data', 'preprocessed_valid{}.npz'.format(postfix)))

    if args.onehot:
        train_inputs = input_onehot(train_data['rgb'], train_data['flow'], train_data['labels'], params)
        valid_inputs = input_onehot(valid_data['rgb'], valid_data['flow'], valid_data['labels'], params)
    else:
        train_inputs = input_fn(train_data['visuals'], train_data['words'], train_data['labels'], params, is_training=True)
        valid_inputs = input_fn(valid_data['visuals'], valid_data['words'], valid_data['labels'], params, is_training=False)

    # Define the model
    logging.info("Building the model")
    if args.onehot:
        train_model_spec = model_onehot(train_inputs, params, mode='train')
        valid_model_spec = model_onehot(valid_inputs, params, mode='valid')
    else:
        train_model_spec = model_fn(train_inputs, params, mode='train')
        valid_model_spec = model_fn(valid_inputs, params, mode='valid')

    # Train the model
    # TODO: Implement for onehot
    logging.info("Training the model")
    if args.onehot:
        training_onehot.train_and_validate_model(train_model_spec, valid_model_spec, model_dir, params)
    else:
        train_and_validate_model(train_model_spec, valid_model_spec, model_dir, params)