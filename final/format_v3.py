import json
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import threading
import multiprocessing

nltk.download('stopwords')
nltk.download('punkt')
rating_map = {0: 'False', 1: 'Partially True', 2: 'True'}


stop_words = set(stopwords.words('english'))

class Pipeline:
    def __init__(self, text, stop_words, trash=[]):
        self.text = text
        self.stop_words = stop_words
        self.redundant = ['#', '$', '%', '&', '\'', '(', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~']
        self.trash = trash

    def is_link(self):
        return 'Link:' in self.text

    def is_related_to_claim(self, claim):
        claim_list = claim.split()
        text_list = self.text.split()

        for c in claim_list:
            for t in text_list:
                if c == t:
                    return True
        return False
    
    def to_lower(self):
        self.text = self.text.lower()

    def remove_weird_characters(self):
        self.text = ''.join([i for i in self.text if i not in self.redundant])

    def remove_trash(self):
        for s in self.trash:
            if s in self.text:
                self.text = self.text.replace(s, '')

    def remove_url(self):
        self.text = re.sub(r'http\S+', '', self.text)

    def remove_extra_space(self):
        self.text = ' '.join(self.text.split())

    def remove_stopwords(self):
        word_tokens = word_tokenize(self.text)
        filtered_sentence = [w for w in word_tokens if w not in self.stop_words]
        self.text = ' '.join(filtered_sentence)

    def run(self):
        self.to_lower()
        self.remove_weird_characters()
        self.remove_trash()
        self.remove_url()
        self.remove_extra_space()
        self.remove_stopwords()


def concat_articles(filenames, claim, trash):
    result = []
    for filename in filenames:
        with open(os.path.join('articles', filename), 'r') as f:
            data = json.load(f)
        if len(data) == 0:
            continue

        for sentence in data:
            p = Pipeline(sentence, stop_words, trash)
            if p.is_link():
                continue

            # claim also need to be preprocessed, but no need to remove redundant string
            claim_p = Pipeline(claim, stop_words)
            claim_p.run()
            p.run()

            # check whether this sentence is related to claim
            if p.is_related_to_claim(claim_p.text):
                result.append(p.text)
    
    return ' '.join(result)


def format_job(data, json_path, i, trash):
    pid = os.getpid()
    local_result = []
    
    tdqm_local_progress = tqdm(data)
    for d in tdqm_local_progress:
        tdqm_local_progress.set_description(f"Process #{i} (Pid:{pid})")
        
        # get question
        question = d['metadata']['claim']

        # get article
        filenames = []
        for key in d['metadata']['premise_articles'].keys():
            filenames.append(d['metadata']['premise_articles'][key])
        article = concat_articles(filenames, question, trash)
        # get options
        options = ['False', 'Partially True', 'True']
        if 'test' not in json_path:
            # get answer
            answer = rating_map[int(d['label']['rating'])]
            local_result.append({
                'id': int(d['metadata']['id']),
                'article': article,
                'question': question,
                'options': options,
                'answer': answer,
            })
        else:
            local_result.append({
                'id': int(d['metadata']['id']),
                'article': article,
                'question': question,
                'options': options,
                'answer': 'None'
            })
    return local_result


def format_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = []
    process_num = 8
    with open('redundant.txt', 'r') as f:
        trash = [line.rstrip() for line in f]
    
    process_data_len = len(data) // process_num
    process_data_split = []
    for i in range(process_num):
        start = i * process_data_len
        end = (i+1) * process_data_len if i != process_num - 1 else len(data)
        process_data_split.append(data[start:end])
    
    pool = multiprocessing.Pool(processes=process_num)
    result_array = pool.starmap(format_job, zip(process_data_split, [json_path]*process_num, range(process_num), [trash]*process_num))

    print("Finished all processes")
        
    for i in range(process_num):
        result.extend(result_array[i])
    
    print("Appended all data")
    
    with open('new_v3_'+json_path, "w") as write_file:
        json.dump(result, write_file)



