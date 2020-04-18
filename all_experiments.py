import json
import os

# Constants 
balance = False
visual_feature_size = 2048
lambda_cross = 1.0
num_epochs = 150
fc1 = 256
num_triplets = 100
word_embedding = "w2v_wiki_100"
prefetch_train = 4
prefetch_test = 4
num_parallel_calls = 4

# Variables
batch_sizes = [256, 128]
lambda_withins = [0.1, 1.0]
learning_rates = [1e-5, 1e-4, 1e-3]
margins = [0.1, 0.5, 1.0, 2.0]
triplet_samplings = ['avg', 'hard']
fc2s = [128, 256]

counter = 1
run_file = open('run_all_experiments.sh', 'w')

for batch_size in batch_sizes:
    for lambda_within in lambda_withins:
        for learning_rate in learning_rates:
            for margin in margins:
                for triplet_sampling in triplet_samplings:
                    for fc2 in fc2s:
                        dir_path = os.path.join('experiments', 'combination_{}'.format(counter))
                        if not os.path.isdir(dir_path):
                            os.makedirs(dir_path)
                        file_path = os.path.join(dir_path, 'params.json')
                        
                        params = {
                            "balance": balance,
                            "visual_feature_size": visual_feature_size,
                            "lambda_cross": lambda_cross,
                            "num_epochs": num_epochs,
                            "fc1": fc1,
                            "num_triplets": num_triplets,
                            "word_embedding": word_embedding,
                            "prefetch_train": prefetch_train,
                            "prefetch_test": prefetch_test,
                            "num_parallel_calls": num_parallel_calls,

                            "batch_size": batch_size,
                            "lambda_within": lambda_within,
                            "learning_rate": learning_rate,
                            "margin": margin,
                            "triplet_sampling": triplet_sampling,
                            "fc2": fc2
                        }

                        json_object = json.dumps(params, indent=4)
                        with open(file_path, 'w') as outfile:
                            outfile.write(json_object)
                        
                        counter += 1
                        run_file.write('python train.py {}/\n'.format(dir_path))
                        run_file.write('python test.py {}/\n\n'.format(dir_path))