# ADL Project
Project for Advanced Deep Learning course, focused on verb classification in videos from EPIC Kitchens dataset using embedding concepts. The work is based on the *Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings* [http://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf](paper).

## The breakdown of the repository tree
The repository has the following directories and subdirectories:
* __baselines__ - contains the baseline action recognition models.
* __data__ - contains the data to be used during training/validation/testing and the scripts to build that data.
    * __embeddings__ - contains dictionaries for embeddings of verb classes. Embeddings are obtained from different word embedding models (e.g., Word2Vec, GloVe).
    * __splits__ - contains the data for training/validation/test sets.
    * __scirpts__ - contains the script files to build the dataset splits and verb embeddings.
* __experiments__ - contains the directories of different experiments.
    * __experiment1__ - contains the weights and log files for *experiment1*, as well as the **params.json** file that has all hyper parameters, including the selected baseline model and word embeddings.
* __model__ - contains the libraries for building/training/evaluation of the model.
* train.py - trains the selected model.
* test.py - evaluates the selected model.

## Task list
- [] - Build dataset from many shot verbs. Use 60-20-20 split.
- [] - Get different word embeddings for the verb classes.
- [] - Come up with the list of hyper parameters for **params.json** files.
- [] - Come up with the list of experiments.
- [] - Process videos into TensorFlow dataset object.
- [] - Build the model.
- [] - Build the evaluation procedure.
- [] - Build the training procedure.
- [] - Train the models.
- [] - Evaluate the models.
- [] - Train the baseline models with one-hot encoding.
- [] - Compare the embedding strategy with one-hot encoding strategy.