"""
STUFF
recall: (text, {entities: [ (start,end,label) ]})
"""

import copy
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

def convert_spacy_to_iob(datapoint):


def convert_spacy_to_iob(datapoint):
    text = copy.deepcopy(datapoint[0])
    entities = copy.deepcopy(datapoint[1]['entities'])

    tokenized = text.split()
    cur_start = 0
    state = 'O'
    tags = []
    for i in range(len(tokenized)):
        if(entities):
            token = tokenized[i]
            cur_start = i
            cur_end = cur_start + len(token)
            print("start:", cur_start, " end:", cur_end, " token:", token)
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
