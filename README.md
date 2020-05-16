# RoBERTa_embeddings
Distill static contextual embeddings from a huggingface transformers RoBERTa model for a specific vocabulary.

## Installation
Install the [huggingface transformers library](https://github.com/huggingface/transformers). Their instructions include guidance about intstalling into a Python virtual environment, which is the recommended approach. An easy way to build a fine-tuned RoBERTa model is to follow their [RoBERTa language modeling example](https://huggingface.co/transformers/examples.html#language-model-training), modified to suit your needs. Be sure to install from source if you want to work with the examples.

Assuming you now have an activated virtual environment, all the other necessary libraries can be installed to it by executing (from within the RoBERTa_embeddings directory):
`pip install -r requirements.txt`

## Other Requirements
### A Model with Hidden States Saved
The Python modules in this repo are meant to be used to create static contextual embeddings from a fine-tuned huggingface transformers RoBERTa masked language model, either the base or large variety. They distill embeddings from one or more of the hidden layers of the fine-tuned model. In order to access those layers, you must output RoBERTa's hidden states when performing the fine-tuning. Hidden layers can be saved by providing a configuration file with output_hidden_states set to true:
`"output_hidden_states": true`
By default, this option is set to false.
You can pass the configuration file to the `run_language_modeling.py` script using this option:

`--config_name=<path/to/config_file>.json`

Example configuration files for base and large RoBERTa for Maked Language Modeling (RobertaForMaskedLM) architectures are given in the top level of the repo as `config_roberta_base.json` and `config_roberta_lg.json`.

### Provide a Vocabulary File
The script in this repo produces embeddings similar to the word embeddings generated by Word2Vec or GloVe, but based on the full context (up to a limit of 512 tokens) in which a word is found. A list of vocabulary words for which embeddings should be generated must be passed in as an argument. Words are split on line breaks, and should be lower case.

### Provide a Context File
The script in this repo distills contextual embeddings for a list of vocabulary words from one or more hidden layers of a RoBERTa masked language model by running up to 2000 (a parameter that can be changed by editing the code) sentences per word through the model. The sentences provide context, resulting in different feature values for the same word. See [this post by Jay Alammar](http://jalammar.github.io/illustrated-transformer/) for a comprehensive description of how transformers work, or read the [RoBERTa paper](https://arxiv.org/pdf/1907.11692.pdf). Unique feature values are calculated for each layer of the model for every unique sentence. If more than one layer is used to create the static contextual embeddings, the layers are summed. A single static contextual embedding for a word is calculated at the end of this process by averaging over all its individual contextual embeddings. Note that by processing the context-specific embeddings in this way, different senses of a word are combined into a single representation.

Sentences are pulled from a text file, the path to which must be provided as an option when calling the script. The script will run much faster if the text file has been filtered on the vocabulary list prior to feeding it in. Sentences are units of text split by line breaks or periods (.). RoBERTa can process a maximum of 512 tokens per sentence, so sentences exceeding this threshold are discarded. Note that the Word Piece tokenization technique employed with BERT/RoBERTa breaks some words into more than one token; there are also seperator tokens that get added at the beginning and end of each sentence. As a result, the word limit per sentence will be <= 510. Multi-token words have their token embeddings During processing, sentences are converted to lowercase for matching against vocabulary words.

## Example Script Invocation
Below is an example of running the main script with all arguments supplied. The files in this example are NOT provided with the repo, so please replace their names with your own.
<pre>python make_contextual_embeddings.py data/output_wiki-103_filtered/ data/wiki.test.raw.out data/FE_vocab_study.txt data/wiki-103_embeddings.txt --count_file data/wiki-103_counts.txt</pre>

To get a description of all required and optional arguments, see the code in `make_contextual_embeddings.py` or (from the command line) execute:
<pre>python make_contextual_embeddings.py --help</pre>