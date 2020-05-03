# RoBERTa_embeddings
Distill static contextual embeddings from a huggingface transformers RoBERTa model for a specific vocabulary.

## Installation
Install the [huggingface transformers library](https://github.com/huggingface/transformers). Their instructions include guidance about intstalling into a Python virtual environment, which is the recommended approach. An easy way to build a fine-tuned RoBERTa model is to follow their [RoBERTa language modeling example](https://huggingface.co/transformers/examples.html#language-model-training), modified to suit your needs. Be sure to install from source if you want to work with the examples.

Assuming you now have an activated virtual environment, all the other necessary libraries can be installed to it by executing (from within the RoBERTa_embeddings directory):
`pip install -r requirements.txt`

## Other Requirements
The scripts in this repo are meant to be used to create static contextual embeddings from a RoBERTa masked language model, fine-tuned on one of the huggingface transformers base models. They distill embeddings from one or more of the hidden layers of the RoBERTa model. In order to access those layers, you must output RoBERTas hidden states when fine-tuning the model. This can be acheived by providing a configuration file with output_hidden_states set to true:
`"output_hidden_states": true`
By default, this option is set to false.
You can pass the configuration file to the `run_language_modeling.py` script using this option:
`--config_name=<path/to/config_file.json`
Example configuration files for base and large RoBERTa for Maked Language Modeling (RobertaForMaskedLM) architectures are given in the top level of the repo.

