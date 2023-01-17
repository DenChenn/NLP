import os
import json
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')


# this will process each row of data
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
        self.remove_weird_characters()
        self.remove_url()
        self.remove_extra_space()
        self.remove_stopwords()


def polish_articles():
    source_dir = 'articles'
    output_dir = 'polished_articles'


    for filename in os.listdir(source_dir):
        print(filename)
        with open(os.path.join(source_dir, filename), 'r') as f:
            data = json.load(f)
        # some articles are empty
        if len(data) == 0:
            continue

        result = []
        for sentence in data:
            p = Pipeline(sentence)
            if p.is_link():
                continue

            claim_p = Pipeline()
            p.run()
            result.append(p.text)

        with open(os.path.join(output_dir, filename), "w") as new_file:
            new_file.write(' '.join(result))


if __name__ == '__main__':
    polish_articles()

