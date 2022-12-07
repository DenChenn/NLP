from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import time
import spacy


P_TRAIN_CSV = 'p_train.csv'
P_VALID_CSV = 'p_val.csv'
P_TEST_CSV = 'p_test.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Preprocess:
    def __init__(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.valid_df = None
        self.test_iter = None
        self.val_iter = None
        self.train_iter = None
        self.input_dim = None
        self.output_dim = None
        self.embedding_dim = None
        self.text = None
        self.label = None

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
        self.text = Field(
            tokenize='spacy',
            lower=True,
            tokenizer_language='en_core_web_sm',
            include_lengths=True
        )
        self.label = LabelField()

        # fields is recognized by order of csv column index, here is first two column
        train, val, test = TabularDataset.splits(
            path='.',
            train=P_TRAIN_CSV,
            validation=P_VALID_CSV,
            test=P_TEST_CSV,
            format='csv',
            fields=[('Utterance', self.text), ('Emotion', self.label)]
        )
        print(vars(train.examples[0]))
        print(vars(val.examples[0]))
        print(vars(test.examples[0]))

        # the model will use the GloVe word vectors trained on:
        # 6 billion words with 100 dimensions per word
        self.text.build_vocab(
            train,
            vectors="glove.6B.100d",
            unk_init=torch.Tensor.normal_
        )
        self.label.build_vocab(train)
        self.embedding_dim = 100
        self.input_dim = len(self.text.vocab)
        print(self.text.vocab.vectors.shape)
        print(f"Unique tokens in TEXT vocabulary: {len(self.text.vocab)}")
        print(f"Unique tokens in LABEL vocabulary: {len(self.label.vocab)}")
        print(self.label.vocab.stoi)
        self.output_dim = len(self.label.vocab)

        # this will return the iterator for each dataset
        # in each dataset, it is sorted by the length of text
        # the device argument is used to specify using the CPU

        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            (train, val, test),
            batch_size=64,
            sort_within_batch=True,
            sort_key=lambda x: len(x.Utterance),
            device=DEVICE)

    def run(self):
        self.select_and_save()
        self.word_embedding()


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.Utterance
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.Emotion)
        acc = categorical_accuracy(predictions, batch.Emotion)
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
            text, text_lengths = batch.Utterance
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.Emotion)
            acc = categorical_accuracy(predictions, batch.Emotion)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


nlp = spacy.load('en_core_web_sm')


def predict_class(text, model, sentence, min_len=4):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += [''] * (min_len - len(tokenized))
    indexed = [text.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor([len(indexed)])
    preds = model(tensor, length_tensor)
    max_preds = preds.argmax(dim=1)
    return max_preds.item()


if __name__ == '__main__':
    pre = Preprocess('train.csv', 'dev.csv')
    pre.run()

    HIDDEN_DIM = 256
    OUTPUT_DIM = pre.output_dim
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = pre.text.vocab.stoi[pre.text.pad_token]
    model = RNN(pre.input_dim,
                pre.embedding_dim,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)

    pretrained_embeddings = pre.text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = pre.text.vocab.stoi[pre.text.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(pre.embedding_dim)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(pre.embedding_dim)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    N_EPOCHS = 10
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, pre.train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, pre.val_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('model.pt'))
    test_loss, test_acc = evaluate(model, pre.test_iter, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    pred = []
    df = pd.read_csv('test.csv')

    for index, row in df.iterrows():
        pred.append(predict_class(pre.text, model, row['Utterance']))

    output_df = pd.DataFrame({'index': [i for i in range(len(pred))], 'emotion': pred})
    output_df.to_csv('output.csv', index=False)
