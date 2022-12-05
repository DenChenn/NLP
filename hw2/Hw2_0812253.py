from torchtext.legacy.data import Field, LabelField, TabularDataset, Iterator
import torch


def read_data():
    text = Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
    label = LabelField(dtype=torch.float)

    train, val, test = TabularDataset.splits(
        path='.', train='train.csv',
        validation='dev.csv', test='test.csv', format='csv',
        fields=[('Text', text), ('Label', label)])

    text.build_vocab(train, vectors="glove.6B.100d")
    train_iter, val_iter, test_iter = Iterator.splits(
        (train, val, test),
        sort_key=lambda x: len(x.Text),
        batch_sizes=(32, 256, 256),
        device=-1
    )
    vocab = text.vocab
    print(vocab.vectors.shape)

    return


if __name__ == '__main__':
    read_data()
