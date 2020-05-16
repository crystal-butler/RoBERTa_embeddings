def make_vocab(vocab_file):
    """Convert a file of newline separated words into a Python list and return it."""
    vocab = []
    with open(vocab_file, 'r') as v:
        vocab = v.read().splitlines()
    return vocab


def tokenize_text(text, tokenizer):
    """Break the input text into tokens the model can use, and return them.
    Use max_length to avoid overflowing the maximum sequence length for the model."""
    tokenized_text = tokenizer.encode(text, add_special_tokens=True, max_length=512)
    return tokenized_text


def print_tokenized_text(tokens, tokenizer):
    """Print the number of tokens in some tokenized text, not counting the leading and trailing separators.
    Print each token without any leading or trailing whitespace."""
    print(f'\nThere are {len(tokens) - 2} tokens in tokenized text:')
    for t in tokens[1:-1]:
        print(tokenizer.decode(t).strip())


def get_vocab_indices(v_tokens, line_tokens, tokenizer):
    """Search a line for all tokens of a vocabulary word, and return the indices of their locations."""
    indices = []              
    for t in v_tokens[1:-1]:
        for i, token_str in enumerate(line_tokens):
            if tokenizer.decode(token_str).strip() == tokenizer.decode(t).strip():
                indices.append(i)
    return indices


def create_token_embeddings(tokenized_text):
    """Convert the model into a more usable format: a tensor of size [<token_count>, <layer_count>, <feature_count>]."""
    input_ids = torch.tensor(tokenized_text).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(input_ids, masked_lm_labels=input_ids)
        encoded_layers = outputs[2]
        token_embeddings = torch.stack(encoded_layers, dim=0)  # Concatenate the tensors for all layers.
        token_embeddings = torch.squeeze(token_embeddings, dim=1)  # Remove the "batches" dimension
        token_embeddings = token_embeddings.permute(1,0,2)  # Rearrange the model dimensions.
        print(f'Size of token embeddings is {token_embeddings.size()}')
        return token_embeddings

def preview_token_embedding(tokenized_text, layer, index, index_list, tokenizer):
    """Print the first 5 feature values from a model layer for tokens at specific line indices."""
    v_index = i % len(tokenized_text[1:-1])
    print(f'{tokenizer.decode(tokenized_text[v_index + 1]).strip()} at index {index_list[index]}: ', \
          f'{layer[index_list[index]][:5].tolist()}')
