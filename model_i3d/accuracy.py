import tensorflow as tf

def compute_predictions(visual_embedding, class_embeddings, labels):
    visual_squared = tf.matmul(visual_embedding, tf.transpose(visual_embedding))
    visual_squared = tf.diag_part(visual_squared)

    class_squared = tf.matmul(class_embeddings, tf.transpose(class_embeddings))
    class_squared = tf.diag_part(class_squared)

    dot_product = tf.matmul(visual_embedding, tf.transpose(class_embeddings))

    distances = tf.expand_dims(visual_squared, 1) - 2.0 * dot_product + tf.expand_dims(class_squared, 0)
    distances = tf.sqrt(distances)

    predictions = tf.argmin(distances, axis=1)

    return predictions


def compute_probabilities(visual_embedding, class_embeddings):
    visual_squared = tf.matmul(visual_embedding, tf.transpose(visual_embedding))
    visual_squared = tf.diag_part(visual_squared)

    class_squared = tf.matmul(class_embeddings, tf.transpose(class_embeddings))
    class_squared = tf.diag_part(class_squared)

    dot_product = tf.matmul(visual_embedding, tf.transpose(class_embeddings))

    distances = tf.expand_dims(visual_squared, 1) - 2.0 * dot_product + tf.expand_dims(class_squared, 0)
    distances = tf.sqrt(distances)

    similarities = 2.0 - distances

    predictions = tf.nn.softmax(similarities)
    
    return predictions


def compute_predictions_similarity(visual_embedding, class_embeddings, labels):
    dot_product = tf.matmul(visual_embedding, tf.transpose(class_embeddings))
    similarity = 1.0 / (1.0 + tf.exp(0.0 - dot_product))

    predictions = tf.argmax(similarity, axis=1)

    return predictions