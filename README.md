# ADL Project
Project for Advanced Deep Learning course, focused on verb classification in videos from EPIC Kitchens dataset using embedding concepts. The work is based on the *Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings* [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf).

## The breakdown of the repository tree
The repository has the following directories and subdirectories:
* __data__ - contains the data to be used during training/validation/testing and the scripts to build that data.
    * __word_embeddings__ - contains dictionaries for embeddings of verb classes. Embeddings are obtained from different word embedding models (e.g., Word2Vec, GloVe).
    * __splits__ - contains the data for training/validation/test sets.
    * __scirpts__ - contains the script files to build the dataset splits and verb embeddings.
* __experiments__ - contains the directories of different experiments.
    * __experiment1__ - contains the weights and log files for *experiment1*, as well as the **params.json** file that has all hyper parameters, including the selected baseline model and word embeddings.
* __model_i3d__ - contains the libraries for building/training/evaluation of the model based on Two Stream I3D.
* __model_epic__ - contains the libraries for building/training/evaluation of the model based on pretrained EPIC KITCHENS model
* train.py - trains the selected model.
* test.py - evaluates the selected model.

## Task list
- [x] - Build dataset from many shot verbs. Use 60-20-20 split.
- [ ] - Get different word embeddings for the verb classes.
    - [x] - 100-dim Word2Vec pretrained on Wikipedia
    - [ ] - 300-dim Word2Vec pretrained on Wikipedia
    - [ ] - 100-dim Word2Vec pretrained on Google News
    - [ ] - GloVe models
- [x] - Get word embeddings for each verb in each video.
- [x] - Come up with the list of hyper parameters for **params.json** files.
- [ ] - Come up with the list of experiments.
- [x] - Process videos into TensorFlow dataset object.
- [x] - Build the model.
- [x] - Build the evaluation procedure.
- [x] - Build the training procedure.
- [ ] - Train the models.
- [ ] - Evaluate the models.
- [ ] - Train the baseline models with one-hot encoding.
- [ ] - Compare the embedding strategy with one-hot encoding strategy.
- [ ] - Think about which metrics to use, can use Precision and Recall along with Accuracy.
- [x] - Precompute the extracted RGB and Flow features from Temporal Segment Networks. 

## How to run training
Copy __visual_features__ directory from Google Drive into __model_epic__ directory. Then run:
```
python train.py path_to_experiment_dir
python test.py path_to_experiment_dir
```

To train and test verb classification with one-hot encoding, run:
```
python train.py -o -p path_to_experiment_dir/[rgb|flow]
python test.py -o -p path_to_experiment_dir
```

## Parameters
* __batch_size__ - Batch size, authors use 256
* __visual_feature_size__ - Size of the visual feature vectors extracted via CNN
* __lambda_within__ and __lambda_cross__ - Weights of the within modal and cross modal triplet losses, authors use 0.1 and 1.0 respectively
* __learning_rate__ - Learning rate, authros use 1e-5
* __margin__ - The marginal constant in the triplet loss, authors didn't mention the value they used
* __num_epochs__ - Number of epochs. Decided to switch from step based to epoch based metrics reading during training
* __num_triplets__ - Number of randomly selected triplets per query during training, authors use 100. __Note:__ we couldn't implement it as tensorflow can't compute gradients for random operations.
* __triplet_sampling__ - Sampling strategy for the triplet losses. Currently, 3 methods were implemented: 
    * __hard__ - Triplets with highest positive sample distances and lowest negative sample distances
    * __total__ - Some up all triplet losses
    * __avg__ - Average of all _valid_ triplet losses
* __fc1__ and __fc2__ - Dimensionality of fully connected layers. Size of FC2 defines the dimensionality of the embedding vectors.
* __word_embedding__ - Method for embedding words. Currently use 100-dimensional Word2Vec trained on Wikipedia