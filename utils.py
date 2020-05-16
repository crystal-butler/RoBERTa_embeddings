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