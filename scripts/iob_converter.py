"""
This file is where I convert spacy's training data format to the BERT-friendly
IOB format.
recall: (text, {entities: [ (start,end,label) ]})
"""

import copy
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
from random_loader import RANDOM_LOADER

class IOB_CONVERTER:
    def convert_spacy_to_iob(datapoint):
        """
        Want: to put a datapoint of the form (text, { 'entities': [(start, end, label)] })
        into the form of a table:
        entry | O
        taxon | B-TAX
        taxon | I-TAX
        """
        text = copy.deepcopy(datapoint[0])
        entities = copy.deepcopy(datapoint[1]['entities'])

        tokenized = text.split()
        cur_start = 0
        state = 'O'
        tags = []
        for i in range(len(tokenized)):
            if(entities):
                token = tokenized[i]
                cur_start = len(" ".join(tokenized[0:i])) + 1
                if cur_start == 1:
                    cur_start = 0
                cur_end = cur_start + len(token)
                if state == "O" and (cur_start <= entities[0][0]) and (entities[0][0]  < cur_end) :
                    tags.append("B-" + entities[0][2])
                    state = "I-" + entities[0][2]
                elif (state.startswith("I-")) and (cur_start < entities[0][1]) and  (entities[0][1] <= cur_end):
                    tags.append(state)
                    state = "O"
                    entities.pop(0)
                else:
                    tags.append(state)
            else :
                tags.append(state)

        return (tokenized, tags)

    def build_csv(data):
        sent_col  = []
        token_col = []
        tags_col  = []
        for i in range(len(data)):
            point = data[i]
            iob_out = IOB_CONVERTER.convert_spacy_to_iob(point)
            tokens = iob_out[0]
            tags = iob_out[1]
            # appending to lists
            sent_col.append(i)
            token_col.append(' '.join(tokens))
            tags_col.append(','.join(tags))
        # Making df
        df = pd.DataFrame({'Sentence': sent_col, 'Words': token_col, 'Tags': tags_col})
        return df
