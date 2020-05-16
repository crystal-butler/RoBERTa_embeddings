import argparse
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM, RobertaConfig
import utils
import distill_layers

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='a directory containing the output from ' \
    'fine-tuning a huggingface RoBERTa model with hidden layers saved', type=str)
parser.add_argument('context_file', help='a text file to use as context to create embeddings', type=str)
parser.add_argument('vocab_file', help='a newline-delimited list of vocabulary words for which ' \
    'to generate embeddings', type=str)
parser.add_argument('output_file', help='path to the file where embeddings should be written', type=str)
parser.add_argument('--count_file', help='optional path to a file where counts of the number ' \
    'of context sentences per vocabulary word should be written', default=None, type=str)
args = parser.parse_args()

tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
model = RobertaForMaskedLM.from_pretrained(args.model_path)
model.eval()

vocab = utils.make_vocab(args.vocab_file)
FEATURE_COUNT = 768  # Change this value to 1024 for the large RoBERTa model.
MAX_LINES = 2000  # Maximum number of context lines to average per vocabulary embedding.


# Process vocabulary words in the outer loop.
for v in vocab:
    start = timer()
    with open(context_file, 'r') as lines:
        v_sum = torch.zeros([1, FEATURE_COUNT])
        v_tokens = tokenize_text(v, tokenizer)
        print_tokenized_text(v_tokens, tokenizer)
        count_sentence = 0
        count_tensor = 0
        
        # Process all lines in the context file in the inner loop.
        for line in lines:
            # Check for this vocab word in this line; if found, split the line into individual sentences.
            if v in line.lower().split():
                for sentence in line.split('.'):
                    if v in sentence.lower():
                        line = sentence
                        count_sentence += 1
                        print(f'\nInstance {count_sentence} of {tokenizer.decode(v_tokens[1:-1]).strip()}.')
                        break  # We'll take the first instance of the word and discard the rest of the line.
                # Split the new sentence-based line into tokens.
                line_tokens = tokenize_text(line, tokenizer)               
                # Get the indices of the line at which our vocabulary word tokens are located.
                indices = get_vocab_indices(v_tokens, line_tokens, tokenizer)                             

                # If the vocabulary word was found, process the containing line.
                if indices:
                    # Get the feature vectors for all tokens in the line/sentence.
                    token_embeddings = create_token_embeddings(line_tokens)
                    # Select a method for extracting specific layers of the model.
                    token_vecs_layer = get_layer_token_vecs(token_embeddings, 12)
                    # Sum the individual token contextual embeddings for the whole vocab word, for this line.
                    tensor_layer = torch.zeros([1, FEATURE_COUNT])
                    for i in range(len(indices)):
                        preview_token_embedding(v_tokens, token_vecs_layer, i, indices, tokenizer)
                        tensor_layer += token_vecs_layer[indices[i]]
                    # If our vocab word is broken into more than one token, we need to get the mean of the token embeddings.
                    tensor_layer /= len(indices)

                    # Add the embedding distilled from this line to the sum of embeddings for all lines.
                    v_sum += tensor_layer
                    count_tensor += 1
                    print(f'Grand sum of {count_tensor} tensor sets is: {v_sum[0][:5].tolist()}')

            # Stop processing lines once we've found 2000 instances of our vocab word.
            if count_tensor >= MAX_LINES:
                break
        
        # We're done processing all lines of 512 tokens or less containing our vocab word.
        # Get the mean embedding for the word.
        v_mean = v_sum / count_tensor
        print(f'Mean of {count_tensor} tensors is: {v_mean[0][:5]} (first 5 of {len(v_mean[0])} features in tensor)')
        write_embedding(output_file, v, v_mean)
        write_line_count(count_file, v, count_tensor)
    end = timer()
    print(f'Run time for {v} was {end - start} seconds.')