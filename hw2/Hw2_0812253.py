from torchtext.legacy.data import Field, LabelField, TabularDataset, Iterator
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import time


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
        self.input_dim = None
        self.embedding_dim = None

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
        label.build_vocab(train)
        self.embedding_dim = 100
        self.input_dim = len(text.vocab)
        print(text.vocab.vectors.shape)
        print(f"Unique tokens in TEXT vocabulary: {len(text.vocab)}")
        print(f"Unique tokens in LABEL vocabulary: {len(label.vocab)}")

        # this will return the iterator for each dataset
        # in each dataset, it is sorted by the length of text
        # the device argument is used to specify using the CPU
        self.train_iter, self.val_iter, self.test_iter = Iterator.splits(
            (train, val, test),
            sort_key=lambda x: len(x.Utterance),
            batch_sizes=(32, 256, 256),
            device=-1
        )

    def run(self):
        self.select_and_save()
        self.word_embedding()


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    # convert into float for division
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.Utterance).squeeze(1)
        loss = criterion(predictions, batch.Emotion)
        acc = binary_accuracy(predictions, batch.Emotion)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.Utterance).squeeze(1)
            loss = criterion(predictions, batch.Emotion)
            acc = binary_accuracy(predictions, batch.Emotion)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    pre = Preprocess('train.csv', 'dev.csv')
    pre.run()

    hidden_dim = 256
    output_dim = 1
    model = RNN(pre.input_dim, pre.embedding_dim, hidden_dim, output_dim)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    N_EPOCHS = 20
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, pre.train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, pre.val_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss, test_acc = evaluate(model, pre.test_iter, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


