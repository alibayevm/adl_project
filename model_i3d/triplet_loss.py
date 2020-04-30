import tensorflow as tf

def get_pairwise_ditance(embeddings_anchor, embeddings, squared):
    dot_product = tf.matmul(embeddings_anchor, tf.transpose(embeddings))

    anchor_squared = tf.matmul(embeddings_anchor, tf.transpose(embeddings_anchor))
    anchor_squared = tf.diag_part(anchor_squared)

    cross_squared = tf.matmul(embeddings, tf.transpose(embeddings))
    cross_squared = tf.diag_part(cross_squared)

    distances = tf.expand_dims(anchor_squared, 1) - 2.0 * dot_product + tf.expand_dims(cross_squared, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        distances = distances * (1.0 - mask)
    
    return distances


def get_cross_modal_mask(labels):
    # Check that j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    distinct_indices = tf.expand_dims(indices_not_equal, 0)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def get_within_modal_mask(labels):
    # Check that i, j, k are distinct
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def get_valid_triplets(labels, embeddings_anchor, embeddings, margin, cross_modal, squared=False):
    """
    Computes and returns a 3D Tensor of losses of all valid triplets. 
    Args:
        labels: labels of the batch with shape (batch_size, )
        embeddings_anchor: batch of anchor embeddings
        embeddings: batch of positive/negative embeddings
        margin: constant margin
        cross_modal: whether embeddings are cross modal or not
    """

    pairwise_distances = get_pairwise_ditance(embeddings_anchor, embeddings, squared)

    positive_dist = tf.expand_dims(pairwise_distances, 2)
    negative_dist = tf.expand_dims(pairwise_distances, 1)

    triplet_loss = positive_dist - negative_dist + margin

    if cross_modal:
        mask = get_cross_modal_mask(labels)
    else:
        mask = get_within_modal_mask(labels)
    mask = tf.to_float(mask)

    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    return triplet_loss


def get_avg_triplet_loss(labels, embeddings_anchor, embeddings, margin, cross_modal, squared=False):
    """
    Computes and returns averaged triplet loss. 
    Args:
        labels: labels of the batch with shape (batch_size, )
        embeddings_anchor: batch of anchor embeddings
        embeddings: batch of positive/negative embeddings
        margin: constant margin
        cross_modal: whether embeddings are cross modal or not
    """
    pairwise_distances = get_pairwise_ditance(embeddings_anchor, embeddings, squared)

    positive_dist = tf.expand_dims(pairwise_distances, 2)
    negative_dist = tf.expand_dims(pairwise_distances, 1)

    triplet_loss = positive_dist - negative_dist + margin

    if cross_modal:
        mask = get_cross_modal_mask(labels)
    else:
        mask = get_within_modal_mask(labels)
    mask = tf.to_float(mask)

    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss


def get_random_triplet_loss(labels, embeddings_anchor, embeddings, margin, num_triplets, cross_modal, squared=False):
    """
    Computes and returns averaged triplet loss. 
    Args:
        labels: labels of the batch with shape (batch_size, )
        embeddings_anchor: batch of anchor embeddings
        embeddings: batch of positive/negative embeddings
        margin: constant margin
        cross_modal: whether embeddings are cross modal or not
    """
    pairwise_distances = get_pairwise_ditance(embeddings_anchor, embeddings, squared)

    positive_dist = tf.expand_dims(pairwise_distances, 2)
    negative_dist = tf.expand_dims(pairwise_distances, 1)

    triplet_loss = positive_dist - negative_dist + margin

    if cross_modal:
        mask = get_cross_modal_mask(labels)
    else:
        mask = get_within_modal_mask(labels)
    mask = tf.to_float(mask)

    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    valid_triplets = tf.greater(triplet_loss, 1e-16)
    triplet_loss = tf.boolean_mask(triplet_loss, valid_triplets)
    # triplet_loss = tf.random_shuffle(triplet_loss)
    triplet_loss = tf.sort(triplet_loss, direction='DESCENDING')
    triplet_loss = tf.concat([triplet_loss, tf.zeros(num_triplets)], 0)
    triplet_loss = tf.slice(triplet_loss, [0], [num_triplets])
    triplet_loss = tf.reduce_sum(triplet_loss)

    return triplet_loss


def get_positive_mask(labels, cross_modal):
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    if not cross_modal:
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        return tf.logical_and(indices_not_equal, labels_equal)

    return labels_equal


def get_negative_mask(labels):
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def batch_hard_triplet_loss(labels, embeddings_anchor, embeddings, margin, cross_modal, squared=False):
    pairwise_distances = get_pairwise_ditance(embeddings_anchor, embeddings, squared)
    
    mask_positive = get_positive_mask(labels, cross_modal)
    mask_positive = tf.to_float(mask_positive)

    positive_dist = tf.multiply(mask_positive, pairwise_distances)

    hardest_positives = tf.reduce_max(positive_dist, axis=1, keepdims=True)

    mask_negative = get_negative_mask(labels)
    mask_negative = tf.to_float(mask_negative)

    max_dist = tf.reduce_max(pairwise_distances, axis=1, keepdims=True)
    negative_dist = pairwise_distances + max_dist * (1.0 - mask_negative)

    hardest_negatives = tf.reduce_min(negative_dist, axis=1, keepdims=True)

    triplet_loss = tf.maximum(hardest_positives - hardest_negatives + margin, 0.0)

    return tf.reduce_mean(triplet_loss)