from nlp import *

def display_dpc_graph(doc):
    displacy.serve(doc, style='dep', port=3000)

data = read_data(TEST_CASE)
nlp = spacy.load('en_core_web_trf')
predict = []
display_dpc_graph(nlp(data.at[28, SENTENCE_COLUMN]))
