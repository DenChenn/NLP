import ast
import re
import pandas as pd
import json
import numpy as np
from transformers import BertTokenizer, TFBertForMultipleChoice
import tensorflow as tf

COLAB_FILE_PREFIX = '/content/drive/MyDrive/hw3/'

def get_answer_key(choice, answer):
    for i in range(len(choice)):
        if answer == choice[i]:
            return i


def format_dataset(json_path):
    format_df = pd.DataFrame(columns=['answerKey', 'article', 'question', 'options'])

    with open(json_path, 'r') as f:
        data = json.load(f)

    not_test_dataset = False
    if 'answer' in data[0][1][0].keys():
        not_test_dataset = True

    for paragraph in data:
        for topic in paragraph[1]:
            options = []
            for choice in topic['choice']:
                options.append(choice)
            if len(options) < 4:
                for i in range(4 - len(options)):
                    options.append(' ')

            if not_test_dataset:
                format_df = format_df.append({
                    'answerKey': get_answer_key(topic['choice'], topic['answer']),
                    'article': paragraph[0][0],
                    'question': topic['question'],
                    'options': options,
                }, ignore_index=True)
            else:
                format_df = format_df.append({
                    'article': paragraph[0][0],
                    'question': topic['question'],
                    'options': options,
                }, ignore_index=True)
    format_df['answerKey'] = pd.to_numeric(format_df['answerKey'])
    return format_df


def preprocessor(dataset, tokenizer):
    """
    This function will convert a given article, question, choices in a format:

    <s> article </s> question </s> choices[0] </s>
    <s> article </s> question </s> choices[1] </s>
    <s> article </s> question </s> choices[2] </s>
    <s> article </s> question </s> choices[3] </s>

    After converting in this format the data will be tokenized using a given tokenizer.
    This function will return 4 arrays namely, input_ids, attention_mask, token_type_ids and labels.

    individual input_ids, token_type_ids, attention_mask shape will be as: [num_choices, max_seq_length]
    """
    all_input_ids = []
    all_attention_mask = []

    for i in range(len(dataset)):
        article = dataset['article'].iloc[i]
        question = dataset['question'].iloc[i]
        options = ast.literal_eval(str(dataset['options'].iloc[i]))
        choice_features = []

        for j in range(len(options)):
            option = options[j]
            input_string = '<s>' + ' ' + article + ' ' + '</s>' + ' ' + question + ' ' + '</s>' + ' ' + option + ' ' + '</s>'
            input_string = re.sub(r'\s+', ' ', input_string)

            input_ids = tokenizer(input_string,
                                  max_length=MAX_LEN,
                                  add_special_tokens=False)['input_ids']

            attention_mask = [1] * len(input_ids)

            padding_id = tokenizer.pad_token_id
            padding_length = 450 - len(input_ids)

            input_ids = input_ids + [padding_id]*padding_length
            attention_mask = attention_mask + [0]*padding_length

            assert len(input_ids) == MAX_LEN
            assert len(attention_mask) == MAX_LEN

            choice_features.append({'input_ids': input_ids,
                                    'attention_mask': attention_mask})

        all_input_ids.append(np.asarray([cf['input_ids'] for cf in choice_features], dtype='int32'))
        all_attention_mask.append(np.asarray([cf['attention_mask'] for cf in choice_features], dtype='int32'))

    return all_input_ids, all_attention_mask


if __name__ == '__main__':
    train_data = format_dataset(COLAB_FILE_PREFIX + 'train_HW3dataset.json')
    dev_data = format_dataset(COLAB_FILE_PREFIX + 'dev_HW3dataset.json')
    test_data = format_dataset(COLAB_FILE_PREFIX + 'test_HW3dataset.json')

    MAX_LEN = 450
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large', do_lower_case=True)

    dev_input_ids, dev_attention_mask = preprocessor(dataset=dev_data, tokenizer=tokenizer)
    dev_input_ids = np.asarray(dev_input_ids, dtype='int32')
    dev_attention_mask = np.asarray(dev_attention_mask, dtype='int32')

    np.save(COLAB_FILE_PREFIX + 'model/race_dev_input_ids', dev_input_ids)
    np.save(COLAB_FILE_PREFIX + 'model/race_dev_attention_mask', dev_attention_mask)

    test_input_ids, test_attention_mask = preprocessor(dataset=test_data, tokenizer=tokenizer)
    test_input_ids = np.asarray(test_input_ids, dtype='int32')
    test_attention_mask = np.asarray(test_attention_mask, dtype='int32')

    np.save(COLAB_FILE_PREFIX + 'model/race_test_input_ids', test_input_ids)
    np.save(COLAB_FILE_PREFIX + 'model/race_test_attention_mask', test_attention_mask)

    train_input_ids, train_attention_mask = preprocessor(dataset=train_data, tokenizer=tokenizer)
    train_input_ids = np.asarray(train_input_ids, dtype='int32')
    train_attention_mask = np.asarray(train_attention_mask, dtype='int32')

    np.save(COLAB_FILE_PREFIX + 'model/race_train_input_ids', train_input_ids)
    np.save(COLAB_FILE_PREFIX + 'model/race_train_attention_mask', train_attention_mask)

    # training
    train_input_ids = np.load(COLAB_FILE_PREFIX + 'model/race_train_input_ids.npy')
    train_attention_mask = np.load(COLAB_FILE_PREFIX + 'model/race_train_attention_mask.npy')

    train_dict = {'input_ids': train_input_ids, 'attention_mask': train_attention_mask}
    train_labels = train_data['answerKey']
    viola = tf.data.Dataset.from_tensor_slices((train_dict, train_labels.values))

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

    strategy = tf.distribute.TPUStrategy(resolver)
    viola = viola.shuffle(32).batch(8).cache().prefetch(tf.data.experimental.AUTOTUNE)

    with strategy.scope():
        model = TFBertForMultipleChoice.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.fit(viola, epochs=4)
    model.save_pretrained(COLAB_FILE_PREFIX + 'model/roberta_model')

    # start predict result for submission
    test_dict = {'input_ids': test_input_ids,
                 'attention_mask': test_attention_mask}
    pred = model.predict(test_dict)
    ans = []
    for log in pred[0]:
        ans.append(np.argmax(log) + 1)
    df = pd.DataFrame({'index': [i for i in range(len(ans))], 'answer': ans})
    df.to_csv(COLAB_FILE_PREFIX+'submission.csv', index=False)
