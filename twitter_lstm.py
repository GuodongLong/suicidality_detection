import sys
import csv
import datetime
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm


def write_auc_csv(data_dict, class_type):
    with open('../output/auc_{}.csv'.format(class_type), 'a') as f:
        field_names = ['date', 'model', 'auc', 'note']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerow(data_dict)


def find_threshold(fpr, tpr, threshold):
    rate = np.array(tpr) + np.array(fpr)
    return threshold[np.argmax(rate)]


def write_metrics_csv(data_dict, class_type):
    with open('../output/eval_{}.csv'.format(class_type), 'a') as f:
        field_names = ['date', 'model', 'accuracy', 'precision', 'recall', 'f-score', 'note', 'dataset']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerow(data_dict)


def predict_evaluate(y_test, y_pred, th, k_th, model_name, subreddit, class_type):
    for i in range(len(y_pred)):
        if y_pred[i] >= th:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    eval = {'date': datetime.date.today(),
            'model': model_name,
            'accuracy': metrics.accuracy_score(y_test, y_pred),
            'precision': metrics.precision_score(y_test, y_pred),
            'recall': metrics.recall_score(y_test, y_pred),
            'f-score': metrics.f1_score(y_test, y_pred),
            'note': '{}_th fold'.format(k_th),
            'dataset': subreddit
            }
    write_metrics_csv(eval, class_type)


EMBEDDING_FILE = '/data/shji/datasets/word2vec/GoogleNews-vectors-negative300.bin'  # replace it with your file path
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
act = 'relu'
print('Loading pertrained embedding file')
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

print('Processing text dataset')
class_type = 'imbalanced'  # considering whether the class is balanced or not
subreddit = 'twitter'  # name of the dataset

file_name = 'sample_data_2.xlsx'  # replace sample data with your input data
df = pd.read_excel(file_name)

# undersampling
num_sampling = 5
for i in range(num_sampling):
    df_pos = df.loc[df['y'] == 1]
    df_neg = df.loc[df['y'] == 0]
    df_sample = pd.concat([df_pos, df_neg.sample(len(df_pos['y']))])

    X_tweets = df_sample['tweets'].as_matrix()
    y = df_sample['y'].as_matrix()
    all_text = []
    for i in range(len(X_tweets)):
        all_text.append(X_tweets[i])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(all_text)

    # 10 fold
    num_fold = 10
    kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)

    for train_index, test_index in tqdm(kf.split(X_tweets)):
        num_fold -= 1
        tweet_train, tweet_test = X_tweets[train_index], X_tweets[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_tweet_seq = tokenizer.texts_to_sequences(tweet_train)
        test_tweet_seq = tokenizer.texts_to_sequences(tweet_test)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        train_tweet = pad_sequences(train_tweet_seq, maxlen=MAX_SEQUENCE_LENGTH)
        test_tweet = pad_sequences(test_tweet_seq, maxlen=MAX_SEQUENCE_LENGTH)
        train_labels = np.array(y_train)
        test_labels = np.array(y_test)
        # print('Shape of data tensor:', train_tweet.shape)
        # print('Shape of label tensor:', train_labels.shape)

        print('Preparing embedding matrix')
        nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        # sample train/validation data
        np.random.seed(1234)
        perm = np.random.permutation(len(train_tweet))
        idx_train = perm[:int(len(train_tweet) * (1 - VALIDATION_SPLIT))]
        idx_val = perm[int(len(train_tweet) * (1 - VALIDATION_SPLIT)):]
        data_train = train_tweet[idx_train]
        labels_train = train_labels[idx_train]
        data_val = train_tweet[idx_val]
        labels_val = train_labels[idx_val]

        # define the model structure
        embedding_layer = Embedding(nb_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = lstm_layer(embedded_sequences)
        merged = Dropout(rate_drop_dense)(x)
        merged = BatchNormalization()(merged)
        merged = Dense(num_dense, activation=act)(merged)
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        # train the model
        model = Model(inputs=sequence_input, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        hist = model.fit(data_train, labels_train,
                         validation_data=(data_val, labels_val),
                         epochs=200, batch_size=2048, shuffle=True,
                         callbacks=[early_stopping])
        # bst_val_score = min(hist.history['val_loss'])

        # predict
        print('Testing')
        preds = model.predict(test_tweet, batch_size=32, verbose=1)
        y_pred = preds.ravel()
        fpr_lstm, tpr_lstm, th_lstm = metrics.roc_curve(y_test, y_pred)
        roc_auc_lstm = metrics.auc(fpr_lstm, tpr_lstm)
        result = {'date': datetime.date.today(),'model': 'LSTM', 'auc':roc_auc_lstm, 'note':'{}_th fold'.format(num_fold)}
        write_auc_csv(result, class_type)
        # o_lstm = find_threshold(fpr_lstm, tpr_lstm, th_lstm)
        o_lstm = 0.5
        predict_evaluate(y_test, y_pred, o_lstm, num_fold, 'LSTM', subreddit, class_type)