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
- [x] - Build dataset from many shot verbs. Use 60-20-20 split. Need to figure out if we need a validation set or not. Technically, we need it for one-hot classifier, but not for the model based on embedding. We could have 80/20 split for train/test, then split train into 60/20 train/valid splits when training one-hot representation.
- [ ] - Get different word embeddings for the verb classes.
- [x] - Get word embeddings for each verb in each video.
- [x] - Come up with the list of hyper parameters for **params.json** files.
- [ ] - Come up with the list of experiments.
- [x] - Process videos into TensorFlow dataset object.
- [x] - Build the model.
- [ ] - Build the evaluation procedure.
- [x] - Build the training procedure.
- [ ] - Train the models.
- [ ] - Evaluate the models.
- [ ] - Train the baseline models with one-hot encoding.
- [ ] - Compare the embedding strategy with one-hot encoding strategy.
- [ ] - Think about which metrics to use, can use Precision and Recall along with Accuracy. Dataset is unbalanced, so 1-h method will favor 4/26 huge classes.
- [x] - Will probably have to precompute the extracted RGB and Flow features from Deep networks due to memory limitations. 

## How to run training
Copy __visual_features__ directory from Google Drive into __model_epic__ directory. Then run:
```
python train.py path_to_experiment_dir
python test.py path_to_experiment_dir
```