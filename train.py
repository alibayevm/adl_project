import tensorflow as tf
import os
import argparse
import logging

# TODO: Consider other models
from model_i3d.utils import Params
from model_i3d.utils import set_logger
from model_i3d.utils import data_info
from model_i3d.input_fn import input_fn
from model_i3d.model_fn import model_fn
from model_i3d.training import train_model



parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Path to the directory with `params.json` file')
args = parser.parse_args()

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
    logging.info("Creating the dataset...")

    train_data = data_info('train')
    train_inputs = input_fn(train_data, params, is_training=True)

    valid_data = data_info('valid')
    valid_inputs = input_fn(valid_data, params, False)

    # Define the model
    logging.info("Building the model")
    train_model_spec = model_fn(train_inputs, params, is_training=True)
    valid_model = model_fn(valid_inputs, params, False)

    # Train the model
    logging.info("Training the model")
    train_model(train_model_spec, valid_model, model_dir, params)