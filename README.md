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
* train.py - trains the selected model.
* test.py - evaluates the selected model.

## How to run training and evaluation
```
python train.py path_to_experiment_dir
python test.py path_to_experiment_dir
```
