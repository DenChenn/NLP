#Author: Yen-Ting Chen
#Student ID: 0812253
#HW ID: 1

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
    print('======')
    print('Accuracy: ', correct/len(answer))
    print('Correct: ', correct)
    print('Wrong Indexes: ' + str(wrong_idx))
    print('======')
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

def get_subject(doc):
    subjects = []
    for token in doc:
        if 'subj' in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

        if 'advmod' == token.dep_ and token.head.pos_ == 'VERB':
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

        if 'pobj' == token.dep_ and token.head.pos_ == 'ADP' and token.head.head.pos_ == 'VERB' and 'agent' == token.head.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

        if 'conj' == token.dep_:
            for child in token.head.children:
                if 'cc' == child.dep_ and child.pos_ == 'CCONJ':
                    subtree = list(token.subtree)
                    start = subtree[0].i
                    end = subtree[-1].i + 1
                    subjects.append(doc[start:end].text)

        if 'nsubj' == token.dep_ and token.head.pos_ == 'VERB' and 'aux' == token.head.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)


    return subjects

def get_object(doc):
    objects = []
    for token in doc:
        if 'dobj' in token.dep_:
            is_not_object = False
            for child in token.children:
                if 'acl' in child.dep_ and child.pos_ == 'VERB':
                    is_not_object = True
                    break
            if is_not_object:
                continue
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            objects.append(doc[start:end].text)
        if 'pobj' == token.dep_ and token.head.pos_ == 'ADP' and token.head.head.pos_ == 'VERB' and 'prep' == token.head.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            objects.append(doc[start:end].text)
        if token.pos_ == 'ADV' and 'advmod' == token.dep_ and token.head.pos_ == 'VERB':
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            objects.append(doc[start:end].text)

        if 'attr' in token.dep_ and token.head.pos_ == 'AUX':
            for child in token.head.children:
                if 'nsubj' in child.dep_:
                    subtree = list(token.subtree)
                    start = subtree[0].i
                    end = subtree[-1].i + 1
                    objects.append(doc[start:end].text)
                    break
    return objects

def is_continuous(doc, start, end):
    for i in range(start, end):
        if doc[i].pos_ != 'ADV' or 'advmod' != doc[i].dep_:
            if doc[i].pos_ != 'AUX' or 'aux' != doc[i].dep_:
                return False
    return True

index_list = []
available_relation = ['aux', 'xcomp', 'prep', 'prt', 'advmod', 'neg', 'auxpass']

def get_index_list(doc):
    for token in doc:
        if token.pos_ == 'VERB':
            get_verb_index(token)
def get_verb_index(token):
    global index_list
    if token.i in index_list:
        return
    index_list.append(token.i)

    for child in token.children:
        if child.dep_ in available_relation:
            get_verb_index(child)
    return

def return_and_reset():
    global index_list
    if len(index_list) == 0:
        return []
    index_list.sort()
    l = index_list.copy()
    index_list = []
    return l


def get_verb(doc):
    verbs = []
    for token in doc:
        if token.pos_ == 'VERB':
            get_verb_index(token)
            indexes = return_and_reset()
            verbs.append(doc[indexes[0]:indexes[-1]+1].text)

        if token.pos_ == 'AUX':
            for child in token.children:
                if 'nsubj' == child.dep_:
                    verbs.append(child.text + ' ' + token.text)

                if 'attr' == child.dep_:
                    for child2 in child.children:
                        if 'prep' == child2.dep_:
                            verbs.append(token.text + ' ' + child.text + ' ' + child2.text)
            if 'auxpass' == token.dep_:
                verbs.append(token.text)

    # return a list of verb string
    return verbs

def get_output(row_data, subjects, verbs, object):
    is_subject = False
    is_object = False
    is_verb = False
    for subj in subjects:
        if row_data[SUBJECT_COLUMN] in subj:
            is_subject = True
            break
    for obj in object:
        if row_data[OBJECT_COLUMN] in obj:
            is_object = True
            break
    for verb in verbs:
        l1 = str(row_data[VERB_COLUMN]).split(' ')
        l2 = verb.split(' ')
        if set(l1).issubset(set(l2)):
            is_verb = True
            break

    return int(is_subject and is_verb and is_object)

def save_answer(predict):
    ans = pd.DataFrame(columns=['index','T/F'])
    ans[INDEX_COLUMN] = [p.row_index for p in predict]
    ans[OUTPUT_COLUMN] = [p.output for p in predict]
    ans.to_csv('predict.csv', index=False)

if __name__ == '__main__':
    # change between formal case and test case
    data = read_data(FORMAL_CASE)
    nlp = spacy.load('en_core_web_trf')
    predict = []

    for i in trange(len(data)):
        sen = data.at[i, SENTENCE_COLUMN]
        subject_list = get_subject(nlp(sen))
        object_list = get_object(nlp(sen))
        verb_list = get_verb(nlp(sen))
        output = get_output(data.iloc[i], subject_list, verb_list, object_list)
        predict.append(PredictResult(output, i, subject_list, verb_list, object_list))

    #validate(data[OUTPUT_COLUMN].tolist(), predict)
    save_answer(predict)
