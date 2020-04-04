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
* __model)i3d__ - contains the libraries for building/training/evaluation of the model based on Two Stream I3D.
* train.py - trains the selected model.
* test.py - evaluates the selected model.

## Task list
- [ ] - Build dataset from many shot verbs. Use 60-20-20 split.
- [ ] - Get different word embeddings for the verb classes.
- [ ] - Get word embeddings for each verb in each video.
- [x] - Come up with the list of hyper parameters for **params.json** files.
- [ ] - Come up with the list of experiments.
- [x] - Process videos into TensorFlow dataset object.
- [x] - Build the model.
- [ ] - Build the evaluation procedure.
- [ ] - Build the training procedure.
- [ ] - Train the models.
- [ ] - Evaluate the models.
- [ ] - Train the baseline models with one-hot encoding.
- [ ] - Compare the embedding strategy with one-hot encoding strategy.
- [ ] - Need to figure out if we need a validation set or not. Technically, we need it for one-hot classifier, but not for the model based on embedding. We could have 80/20 split for train/test, then split train into 60/20 train/valid splits when training one-hot representation.
- [ ] - Think about which metrics to use, can use Precision and Recall along with Accuracy. Dataset is unbalanced, so 1-h method will favor 4/26 huge classes.
