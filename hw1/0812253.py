#Author: Yen-Ting Chen
#Student ID: 0812253
#HW ID: 1

from prettytable import PrettyTable
import pandas as pd
import spacy
from spacy import displacy
from tqdm import trange

INDEX_COLUMN = 'index'
OUTPUT_COLUMN = 'T/F'
SENTENCE_COLUMN = 'Sentence'
SUBJECT_COLUMN = 'S'
VERB_COLUMN = 'V'
OBJECT_COLUMN = 'O'
TEST_CASE = 'example_with_answer.csv'
FORMAL_CASE = 'dataset.csv'

class PredictResult:
    def __init__(self, output, row_index, predicted_subject, predicted_verb, predicted_object):
        self.output = output
        self.row_index = row_index
        self.predicted_subject = predicted_subject
        self.predicted_verb = predicted_verb
        self.predicted_object = predicted_object

def validate(answer, predict):
    if len(answer) != len(predict):
        print('The length of answer and predict are not equal.')
        return

    correct = 0
    wrong = []
    wrong_idx = []
    for i in range(len(answer)):
        if answer[i] == predict[i].output:
            correct += 1
        else:
            wrong.append(predict[i])
            wrong_idx.append(i)
    print("======")
    print('Accuracy: ', correct/len(answer))
    print('Correct: ', correct)
    print('Wrong Indexes: ' + str(wrong_idx))
    print("======")
    for w in wrong:
        print('Row index: ', w.row_index)
        print('Output: ', w.output)
        print('Predicted subject: ', w.predicted_subject)
        print('Predicted verb: ', w.predicted_verb)
        print('Predicted object: ', w.predicted_object)
        print('======')


def read_data(filepath):
    if filepath == TEST_CASE:
        df = pd.read_csv(filepath, names=[OUTPUT_COLUMN, SENTENCE_COLUMN, SUBJECT_COLUMN, VERB_COLUMN,
                                            OBJECT_COLUMN], header=0)
        df[INDEX_COLUMN] = range(0, len(df))
        return df
    elif filepath == FORMAL_CASE:
        df = pd.read_csv(filepath, names=[INDEX_COLUMN, SENTENCE_COLUMN, SUBJECT_COLUMN, VERB_COLUMN,
                                            OBJECT_COLUMN], header=None)
        return df

def display_dpc_graph(doc):
    displacy.serve(doc, style='dep', port=3000)

def get_subject(doc):
    subjects = []
    for token in doc:
        if 'subj' in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

        if 'advmod' in token.dep_ and token.head.pos_ == 'VERB':
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

        if 'pobj' in token.dep_ and token.head.pos_ == 'ADP' and token.head.head.pos_ == 'VERB' and 'agent' in token.head.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

    return subjects

def get_object(doc):
    objects = []
    for token in doc:
        if "dobj" in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            objects.append(doc[start:end].text)
    return objects

def get_verb(doc):
    indexes = []
    for token in doc:
        if token.pos_ == 'VERB':
            indexes.append(token.i)
    # return a list of verb string
    return [doc[i].text for i in indexes]

def get_output(row_data, subjects, verbs, object):
    is_subject = False
    is_object = False
    for subj in subjects:
        if row_data[SUBJECT_COLUMN] in subj:
            is_subject = True
            break
    for obj in object:
        if row_data[OBJECT_COLUMN] in obj:
            is_object = True
            break

    return int(is_subject and row_data[VERB_COLUMN] in verbs and is_object)

def save_answer(predict):
    ans = pd.DataFrame(columns=["index","T/F"])
    ans[INDEX_COLUMN] = [p.row_index for p in predict]
    ans[OUTPUT_COLUMN] = [p.output for p in predict]
    ans.to_csv('predict.csv', index=False)

if __name__ == "__main__":
    data = read_data(TEST_CASE)
    nlp = spacy.load('en_core_web_trf')
    predict = []

    for i in trange(len(data)):
        doc = data.at[i, SENTENCE_COLUMN]
        subject_list = get_subject(nlp(doc))
        object_list = get_object(nlp(doc))
        verb_list = get_verb(nlp(doc))
        output = get_output(data.iloc[i], subject_list, verb_list, object_list)
        predict.append(PredictResult(output, i, subject_list, verb_list, object_list))

    validate(data[OUTPUT_COLUMN].tolist(), predict)
    #display_dpc_graph(nlp(data.at[6, SENTENCE_COLUMN]))
    #save_answer(predict)

