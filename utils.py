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