import tensorflow as tf
import numpy as np
import os
import argparse
import logging
from tqdm import trange

from model_epic.utils import Params
from model_epic.utils import set_logger
from model_epic.utils import save_dict_to_json
from model_epic.data_preprocessing import preprocess_test
from model_epic.input_fn import input_fn_test, input_onehot
from model_epic.model_fn import model_fn, model_onehot

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
    set_logger(os.path.join(model_dir, 'test.log'))

    # Input data pipeline
    if not os.path.isfile(os.path.join('data', 'preprocessed_test.npz')) or args.preprocess:
        visual_rgb = os.path.join('model_epic', 'visual_features', 'test', 'tsn_resnet50_rgb.npy')
        visual_flow = os.path.join('model_epic', 'visual_features', 'test', 'tsn_resnet50_flow.npy')
        split = os.path.join('data', 'splits', 'test.txt')
        preprocess_test(visual_rgb, visual_flow, split, onehot=args.onehot)

    logging.info("Creating the dataset...")
    
    postfix = '_onehot' if args.onehot else ''
    test_data = np.load(os.path.join('data', 'preprocessed_test{}.npz'.format(postfix)))
    
    words = os.path.join('data', 'word_embeddings', '{}_classkeys.npy'.format(params.word_embedding))
    words_classkeys = np.load(words)

    if args.onehot:
        test_inputs = input_onehot(test_data['rgb'], test_data['flow'], test_data['labels'], params)
        test_model_spec = model_onehot(test_inputs, params, 'test')
    else:
        test_inputs = input_fn_test(test_data['visuals'], test_data['labels'], words_classkeys, params)
        test_model_spec = model_fn(test_inputs, params, 'test')

    
    # Start testing
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Restore weights
        if args.onehot:
            
            saver_rgb = tf.train.Saver(var_list=test_model_spec['variable_map_rgb'], reshape=True)
            save_path = tf.train.latest_checkpoint(os.path.join(model_dir, 'rgb', 'last_weights'))
            saver_rgb.restore(sess, save_path)
            saver_flow = tf.train.Saver(var_list=test_model_spec['variable_map_flow'], reshape=True)
            save_path = tf.train.latest_checkpoint(os.path.join(model_dir, 'flow', 'last_weights'))
            saver_flow.restore(sess, save_path)
        else:
            # Initialize saver
            saver = tf.train.Saver()
            save_path = tf.train.latest_checkpoint(os.path.join(model_dir, 'last_weights'))
            saver.restore(sess, save_path)

        update_metrics = test_model_spec['update_metrics']
        metrics = test_model_spec['metrics']

        sess.run(test_model_spec['iterator_init_op'])
        sess.run(test_model_spec['metrics_init_op'])

        t = trange(test_model_spec['num_steps'])
        for _ in t:
            sess.run(update_metrics)
        
        metrics_values = {k: v[0] for k, v in metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Test metrics: " + metrics_string)

        save_dict_to_json(metrics_val, os.path.join(model_dir, 'test_metrics.json'))



