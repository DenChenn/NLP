import pandas as pd

reverse_map = {0: 'neutral', 1: 'joy', 2: 'anger', 3: 'surprise', 4: 'sadness', 5: 'disgust', 6: 'fear'}
formal_index_map = {'neutral': 0, 'anger': 1, 'joy': 2, 'surprise': 3, 'sadness': 4, 'disgust': 5, 'fear': 6}

df = pd.read_csv('output.csv')
df['emotion'] = df['emotion'].map(reverse_map).map(formal_index_map)
df.to_csv('output2.csv', index=False)
