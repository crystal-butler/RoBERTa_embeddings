import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM, RobertaConfig


def sum_last_four_token_vecs(token_embeddings):
    """Sum the last 4 layers' features and return the resulting vector."""
    token_vecs_sum_last_four = []

    # For each token in the sentence, sum the last 4 layers of the model.
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum_last_four.append(sum_vec)

    print ('Shape of summed layers is: %d x %d' % (len(token_vecs_sum_last_four), len(token_vecs_sum_last_four[0])))
    return token_vecs_sum_last_four


def get_layer_token_vecs(token_embeddings, layer_number):
    """Return a single layer of the model."""
    token_vecs_layer = []

    # For each token in the sentence...
    for token in token_embeddings:
        layer_vec = token[layer_number]

        # Use `layer_vec` to represent `token`.
        token_vecs_layer.append(layer_vec)

    print ('Shape of summed layers is: %d x %d' % (len(token_vecs_layer), len(token_vecs_layer[0])))
    return token_vecs_layer
