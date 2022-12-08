from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import spacy
import torch.nn.functional as F


P_TRAIN_CSV = 'p_train.csv'
P_VALID_CSV = 'p_val.csv'
P_TEST_CSV = 'p_test.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NLP = spacy.load('en_core_web_sm')
FORMAL_INDEX_MAP = {'neutral': 0, 'anger': 1, 'joy': 2, 'surprise': 3, 'sadness': 4, 'disgust': 5, 'fear': 6}


class Preprocess:
    def __init__(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.val_df = None
        self.test_iter = None
        self.val_iter = None
        self.train_iter = None
        self.input_dim = None
        self.output_dim = None
        self.embedding_dim = None
        self.text = None
        self.label = None
        self.label_dict = None

    def select_and_save(self):
        self.train_df = self.train_df[['Utterance', 'Emotion']]
        self.test_df = self.test_df[['Utterance', 'Emotion']]
        # separate training set to training set and validation set
        self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.1)
        self.train_df.to_csv(P_TRAIN_CSV, header=False, index=False)
        self.test_df.to_csv(P_TEST_CSV, header=False, index=False)
        self.val_df.to_csv(P_VALID_CSV, header=False, index=False)
        print(f'Size of training set:  {len(self.train_df)}')
        print(f'Size of validation set:  {len(self.val_df)}')
        print(f'Size of testing set:  {len(self.test_df)}')

    def word_embedding(self):
        self.text = Field(
            sequential=True,
            use_vocab=True,
            tokenizer_language='en_core_web_sm',
            tokenize='spacy',
            lower=True, batch_first=True)
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
        print(f'Training set example: {vars(train.examples[0])}')
        print(f'Validation set example: {vars(val.examples[0])}')
        print(f'Testing set example: {vars(test.examples[0])}')

        # the model will use the GloVe word vectors trained on:
        # 6 billion words with 100 dimensions per word
        self.text.build_vocab(
            train,
            vectors='glove.6B.100d',
            unk_init=torch.Tensor.normal_
        )
        self.label.build_vocab(train)
        self.embedding_dim = 100
        self.input_dim = len(self.text.vocab)
        self.output_dim = len(self.label.vocab)
        self.label_dict = dict(self.label.vocab.stoi)
        print(f'Unique tokens in TEXT vocabulary: {len(self.text.vocab)}')
        print(f'Unique tokens in LABEL vocabulary: {len(self.label.vocab)}')

        # create iterator
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            (train, val, test),
            batch_size=64,
            sort_key=lambda x: len(x.Utterance),
            shuffle=True,
            device=DEVICE)

    def run(self):
        self.select_and_save()
        self.word_embedding()


class BidirectionalRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim // 2,
                           num_layers=n_layers,
                           bidirectional=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)

        attn_output = self.attention(output, hidden)
        return self.fc(attn_output.squeeze(0))


def accuracy(preds, y):
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
        batch.Utterance = batch.Utterance.permute(1, 0)
        predictions = model(batch.Utterance).squeeze(1)
        loss = criterion(predictions, batch.Emotion)
        acc = accuracy(predictions, batch.Emotion)
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
            batch.Utterance = batch.Utterance.permute(1, 0)
            predictions = model(batch.Utterance).squeeze(1)
            loss = criterion(predictions, batch.Emotion)
            acc = accuracy(predictions, batch.Emotion)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict_class(text, model, sentence):
    model.eval()
    tokenized = [tok.text for tok in NLP.tokenizer(sentence)]
    indexed = [text.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(1)
    tensor = torch.cat(2*[tensor], dim=1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1)
    return max_preds[0].item()


if __name__ == '__main__':
    pre = Preprocess('train.csv', 'dev.csv')
    pre.run()

    padding_index = pre.text.vocab.stoi[pre.text.pad_token]
    model = BidirectionalRNN(pre.input_dim, pre.embedding_dim, 256, pre.output_dim, 1, 0.5)

    model.embedding.weight.data.copy_(pre.text.vocab.vectors)
    unk_idx = pre.text.vocab.stoi[pre.text.unk_token]
    model.embedding.weight.data[unk_idx] = torch.zeros(pre.embedding_dim)
    model.embedding.weight.data[padding_index] = torch.zeros(pre.embedding_dim)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # start training
    epoch = 15
    best_valid_loss = float('inf')
    for epoch in range(epoch):
        train_loss, train_acc = train(model, pre.train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, pre.val_iter, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        print('------')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Validation Loss: {valid_loss:.3f} |  Validation Acc: {valid_acc*100:.2f}%')
        print('------')

    model.load_state_dict(torch.load('model.pt'))
    test_loss, test_acc = evaluate(model, pre.test_iter, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    # apply on given testing dataset
    pred = []
    df = pd.read_csv('test.csv')
    for index, row in df.iterrows():
        pred.append(predict_class(pre.text, model, row['Utterance']))

    inv_map = {v: k for k, v in pre.label_dict.items()}
    # pack as required format
    output_df = pd.DataFrame({'index': [i for i in range(len(pred))], 'emotion': pred})
    output_df['emotion'] = output_df['emotion'].map(inv_map).map(FORMAL_INDEX_MAP)
    output_df.to_csv('output.csv', index=False)
