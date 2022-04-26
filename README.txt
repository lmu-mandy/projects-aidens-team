Aiden Meyer's TAXONOMY NER MODELS

Greetings! Welcome to my NER model trainer and tester. In this repository, you will notice several folders:
  1. data - this folder contains all data files used to train the named entity recognizer models.
  2. scripts - this folder contains all python scripts that were used to train models, as well as ones that
     demonstrate their capabilities.
  3. taxon_ner_model - this folder contains the trained SpaCy model
  4. bert_taxon_model - this folder contains the trained BERT model (INCOMPLETE)

REQUIREMENTS:
These python scripts require the following modules:
  numpy
  pandas
  spacy
  sklearn
  torch
  transformers

USAGE:
In order to try out the pre-trained models, simply modify the python scripts as such:
  1. In spacy_trainer.py, uncomment 'demo()' at the bottom and comment out
     'train_spacy()'. This will train a fresh NER model and save it to the disk.
     If these lines are already commented appropriately, do nothing.
  2. In bert_trainer, follow the same procedure above (replacing train_spacy with
     train_bert).

In order to train fresh models, comment out the 'demo()' lines and uncomment the 'train_spacy()'
lines.
