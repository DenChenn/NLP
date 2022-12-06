from torchtext.legacy.data import Field, LabelField, TabularDataset, Iterator
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

P_TRAIN_CSV = 'p_train.csv'
P_VALID_CSV = 'p_val.csv'
P_TEST_CSV = 'p_test.csv'


class Preprocess:
    def __init__(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.valid_df = None
        self.test_iter = None
        self.val_iter = None
        self.train_iter = None

    def select_and_save(self):
        self.train_df = self.train_df[['Utterance', 'Emotion']]
        self.test_df = self.test_df[['Utterance', 'Emotion']]
        # separate training set to training set and validation set
        self.train_df, self.valid_df = train_test_split(self.train_df, test_size=0.1)
        self.train_df.to_csv(P_TRAIN_CSV, header=False, index=False)
        self.test_df.to_csv(P_TEST_CSV, header=False, index=False)
        self.valid_df.to_csv(P_VALID_CSV, header=False, index=False)
        print(len(self.train_df))
        print(len(self.valid_df))
        print(len(self.test_df))

    def word_embedding(self):
        text = Field(tokenize='spacy', lower=True, tokenizer_language='en_core_web_sm')
        label = LabelField(dtype=torch.float)

        # fields is recognized by order of csv column index, here is first two column
        train, val, test = TabularDataset.splits(
            path='.',
            train=P_TRAIN_CSV,
            validation=P_VALID_CSV,
            test=P_TEST_CSV,
            format='csv',
            fields=[('Utterance', text), ('Emotion', label)]
        )
        print(vars(train.examples[0]))
        print(vars(val.examples[0]))
        print(vars(test.examples[0]))

        # the model will use the GloVe word vectors trained on:
        # 6 billion words with 100 dimensions per word
        text.build_vocab(train, vectors="glove.6B.100d")

        # this will return the iterator for each dataset
        # in each dataset, it is sorted by the length of text
        # the device argument is used to specify using the CPU
        self.train_iter, self.val_iter, self.test_iter = Iterator.splits(
            (train, val, test),
            sort_key=lambda x: len(x.Text),
            batch_sizes=(32, 256, 256),
            device=-1
        )

    def run(self):
        self.select_and_save()
        self.word_embedding()


if __name__ == '__main__':
    pre = Preprocess('train.csv', 'dev.csv')
    pre.run()
