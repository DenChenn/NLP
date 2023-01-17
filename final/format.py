import json
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')
rating_map = {0: 'False', 1: 'Partially True', 2: 'True'}


class Pipeline:
    def __init__(self, text):
        self.text = text
        self.stop_words = set(stopwords.words('english'))
        self.redundant = ['#', '$', '%', '&', '\'', '(', ')', '*', '+', '-', '/', '<', '=', '>', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~']

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
        self.remove_url()
        self.remove_extra_space()
        self.remove_stopwords()


def concat_articles(filenames, claim):
    result = []
    for filename in filenames:
        with open(os.path.join('articles', filename), 'r') as f:
            data = json.load(f)
        if len(data) == 0:
            continue

        for sentence in data:
            p = Pipeline(sentence)
            if p.is_link():
                continue

            # claim also need to be preprocessed
            claim_p = Pipeline(claim)
            claim_p.run()
            p.run()

            # check whether this sentence is related to claim
            if p.is_related_to_claim(claim_p.text):
                result.append(p.text)
    
    return ' '.join(result)


def format_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = []
    count = 0
    for d in data:
        # get question
        question = d['metadata']['claim']

        # get article
        filenames = []
        count += 1
        print(count)
        for key in d['metadata']['premise_articles'].keys():
            filenames.append(d['metadata']['premise_articles'][key])
        article = concat_articles(filenames, question)
        # get options
        options = ['False', 'Partially True', 'True']

        if 'test' not in json_path:
            # get answer
            answer = rating_map[int(d['label']['rating'])]
            result.append({
                'id': int(d['metadata']['id']),
                'article': article,
                'question': question,
                'options': options,
                'answer': answer,
            })
        else:
            result.append({
                'id': int(d['metadata']['id']),
                'article': article,
                'question': question,
                'options': options,
                'answer': 'None'
            })

    with open('new_'+json_path, "w") as write_file:
        json.dump(result, write_file)

