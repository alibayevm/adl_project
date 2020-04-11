import tensorflow as tf
import numpy as np
import os
import argparse
import logging

from model_epic.utils import Params
from model_epic.utils import set_logger
from model_epic.utils import data_info
from model_epic.data_preprocessing import preprocess
from model_epic.input_fn import input_fn
from model_epic.model_fn import model_fn
from model_epic.training import train_model

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Path to the directory with `params.json` file')
parser.add_argument('-p', '--preprocess', action='store_true', help='Whether to preprocess input data or not')
args = parser.parse_args()

if __name__ == "__main__":
    tf.set_random_seed(230)

    model_dir = args.model_dir

    # Set parameters
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))

    # Input data pipeline
    # TODO: Make it an argument
    if not os.path.isfile(os.path.join('data', 'preprocessed.npz')) or args.preprocess:
        visual_rgb = os.path.join('model_epic', 'visual_features', 'tsn_resnet50_rgb.npy')
        visual_flow = os.path.join('model_epic', 'visual_features', 'tsn_resnet50_flow.npy')
        words = os.path.join('data', 'word_embeddings', '{}_train.npy'.format(params.word_embedding))
        split = os.path.join('data', 'splits', 'train_embedding.npy')
        preprocess(visual_rgb, visual_flow, words, split)

    logging.info("Creating the dataset...")

    train_data = np.load(os.path.join('data', 'preprocessed.npz'))
    train_inputs = input_fn(train_data['visuals'], train_data['words'], train_data['labels'], params, is_training=True)

    # Define the model
    logging.info("Building the model")
    train_model_spec = model_fn(train_inputs, params, is_training=True)

    # Train the model
    logging.info("Training the model")
    train_model(train_model_spec, model_dir, params)