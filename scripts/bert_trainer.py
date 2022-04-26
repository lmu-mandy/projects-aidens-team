""" ============================================================================
INCOMPLETE

BERT Trainer
This is an executable script which trains a fresh spacy model for named entity
recognition, using BERT instead of SpaCy. The model's weights are saved to a
folder named 'bert_taxon_model'

Methods:
    train_spacy     Trains an NER model for a given number of iterations. Saves
                    the model to a folder 'taxon_ner_model'
    test            Tests our pretrained 'taxon_ner_model' and gives us
                    accuracy, precision, and recall scores.
    demo            Prints out some example named entities in some sentences.
============================================================================ """
import copy
import pandas as pd
import numpy as np
import spacy
from spacy.tokens import DocBin
from iob_converter import IOB_CONVERTER
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification

LABEL_TO_ID = {'O': 0, 'B-TAX': 1, 'I-TAX': 2}
ID_TO_LABEL = {0: 'O', 1: 'B-TAX', 2: 'I-TAX'}

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class dataset(Dataset):
    """
    This class was directly taken from:
    https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=Eh3ckSO0YMZW
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.sentence[index].strip().split()
        word_labels = self.data.word_labels[index].split(",")

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                            is_pretokenized=True,
                            return_offsets_mapping=True,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                encoded_labels[idx] = labels[i]
                i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len
