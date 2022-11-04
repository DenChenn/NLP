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
    def __init__(self, output, row_index, data, patches):
        self.output = output
        self.row_index = row_index
        self.data = data
        self.patches = patches

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
        print('Expected Verb: "', w.data[VERB_COLUMN], '"')
        print('Expected Subject: "', w.data[SUBJECT_COLUMN], '"')
        print('Expected Object: "', w.data[OBJECT_COLUMN], '"')
        print('Predicted patches: ')
        for p in w.patches:
            print(p)
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

def get_subject(doc, verb_token):
    subjects = []

    # 關係子句
    if 'relcl' in verb_token.dep_:
        subtree = list(verb_token.head.subtree)
        start = subtree[0].i
        end = subtree[-1].i + 1
        subjects.append(doc[start:end].text)

    # 連接詞後的子句
    if 'acl' in verb_token.dep_:
        subtree = list(verb_token.head.subtree)
        start = subtree[0].i
        end = subtree[-1].i + 1
        subjects.append(doc[start:end].text)

    for token in doc:
        if 'subj' in token.dep_ and token.head.i == verb_token.i:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

        if 'advmod' == token.dep_ and token.head.i == verb_token.i:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

        # passive verb's subject
        if 'pobj' == token.dep_ and token.head.pos_ == 'ADP' and token.head.head.i == verb_token.i and 'agent' == token.head.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subjects.append(doc[start:end].text)

    return subjects

def get_object(doc, verb_token):
    objects = []
    for token in doc:
        if 'obj' in token.dep_ and token.head.i == verb_token.i:
            # special case : "have trouble telling" -> trouble is the object
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

        # 動詞片語後面的也有可能是受詞
        if 'pobj' == token.dep_ and token.head.pos_ == 'ADP' and token.head.head.i == verb_token.i and 'prep' == token.head.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            objects.append(doc[start:end].text)

        if token.pos_ == 'ADV' and 'advmod' == token.dep_ and token.head.i == verb_token.i:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            objects.append(doc[start:end].text)

        if 'attr' in token.dep_ and token.head.i == verb_token.i:
            for child in token.head.children:
                if 'nsubj' in child.dep_:
                    subtree = list(token.subtree)
                    start = subtree[0].i
                    end = subtree[-1].i + 1
                    objects.append(doc[start:end].text)
    return objects


verb_index_list = []
verb_relation = ['aux', 'xcomp', 'prep', 'prt', 'advmod', 'neg', 'auxpass', 'nsubj', 'attr']

def get_verb_index(token, branch, is_edge):
    branch.append(token.i)

    for child in token.children:
        if child.dep_ in verb_relation:
            is_edge = False
            get_verb_index(child, branch.copy(), True)

    if is_edge:
        global verb_index_list
        branch.sort()
        verb_index_list.append(branch)

    return

def return_and_reset():
    global verb_index_list
    if len(verb_index_list) == 0:
        return []
    #verb_index_list.sort()
    l = verb_index_list.copy()
    verb_index_list = []
    return l


def distribute(list_of_index):
    flatten = []
    for l in list_of_index:
        for idx in l:
            if idx not in flatten:
                flatten.append(idx)

    flatten.sort()
    temp = [flatten[0]]
    result = []
    for k in range(1, len(flatten)):
        if flatten[k] - flatten[k-1] == 1:
            temp.append(flatten[k])
        else:
            result.append(temp)
            temp = [flatten[k]]
    result.append(temp)
    return result

def concat(doc, list_of_index):
    return ' '.join([doc[x].text for x in list_of_index])

def get_patch(doc):
    # [["verb", "subject", "object"]]
    patches = []

    for token in doc:
        children_deps = list(token.dep_ for token in token.children)
        if (token.pos_ == 'VERB') or (token.pos_ == 'AUX' and 'relcl' == token.dep_) or (token.pos_ == 'AUX' and 'attr' in children_deps):
            # 1. find verb
            get_verb_index(token, [], True)
            seq_set = return_and_reset()

            verbs = []
            for seq in seq_set:
                verbs.append(concat(doc, seq))

            continuous_seq_set = distribute(seq_set)
            for seq in continuous_seq_set:
                verbs.append(concat(doc, seq))

            # find subject related to this verb
            subjects = get_subject(doc, token)

            # find object related to this verb
            objects = get_object(doc, token)

            # push into list of patches
            for verb in verbs:
                if len(subjects) == 0:
                    patches.append([verb, '', ''])
                    continue
                for sub in subjects:
                    if len(objects) == 0:
                        patches.append([verb, sub, ''])
                        continue
                    for obj in objects:
                        patches.append([verb, sub, obj])

    return patches

def is_subset(s1, s2):
    l1 = str(s1).split(' ')
    l2 = str(s2).split(' ')
    if set(l1).issubset(set(l2)):
        return True
    return False

def get_output(row_data, patches):
    for p in patches:
        is_subject = False
        is_object = False
        is_verb = False
        if is_subset(row_data[VERB_COLUMN], p[0]):
            is_verb = True
        if is_subset(row_data[SUBJECT_COLUMN], p[1]):
            is_subject = True
        if is_subset(row_data[OBJECT_COLUMN], p[2]):
            is_object = True

        if is_verb and is_subject and is_object:
            return 1
    return 0

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
        patches = get_patch(nlp(sen))
        output = get_output(data.iloc[i], patches)
        predict.append(PredictResult(output, i, data.iloc[i], patches))

    #validate(data[OUTPUT_COLUMN].tolist(), predict)
    save_answer(predict)
