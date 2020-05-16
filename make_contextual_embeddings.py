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


