import string
import csv
import nltk
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
from sklearn import metrics
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.svm import SVC


def f_basic(data):
    num_title_words, num_title_token, num_title_char, num_title_sent = [], [], [], []
    num_body_words, num_body_token, num_body_para, num_body_sent = [], [], [], []

    for title in data['title']:
        num_title_words.append(len(title.split()))
        tokens = nltk.word_tokenize(title)
        num_title_token.append(len(tokens))
        num_title_char.append(len(title))
        sentences = nltk.tokenize.sent_tokenize(title, language='english')
        num_title_sent.append(len(sentences))

    for body in data['usertext']:
        temp_words, temp_token, temp_sent = 0, 0, 0
        for para in body:
            temp_words += len(para.split())
            temp_token += len(nltk.word_tokenize(para))
            temp_sent += len(nltk.tokenize.sent_tokenize(para, language='english'))
        num_body_words.append(temp_words)
        num_body_token.append(temp_token)
        num_body_sent.append(temp_sent)
        num_body_para.append(len(body))

    features = {
        'title_words': num_title_words,
        'title_token': num_title_token,
        'title_char': num_title_char,
        'title_sent': num_title_sent,
        'body_words': num_body_words,
        'body_token': num_body_token,
        'body_sent': num_body_sent,
        'body_para': num_body_para
    }
    return pd.DataFrame(features, columns=['title_words', 'title_token', 'title_char', 'title_sent',
                                           'body_words', 'body_token', 'body_sent', 'body_para'])


def f_liwc(class_type, subreddit):
    # replace extracted features using LIWC with your own
    liwc_title = pd.read_csv('sample_1_liwc_title.csv'.format(class_type, subreddit))
    liwc_body = pd.read_csv('sample_1_liwc_body.csv'.format(class_type, subreddit))
    liwc = pd.concat((liwc_title[liwc_title.columns[4:]], liwc_body[liwc_body.columns[4:]]), axis=1)
    return liwc


def get_all_tags(data):
    tags_all = []
    for title in data['title']:
        tagged_text = nltk.pos_tag(nltk.word_tokenize(title))
        for word, tag in tagged_text:
            if tag not in tags_all:
                tags_all.append(tag)
    for body in data['usertext']:
        for para in body:
            tagged_text = nltk.pos_tag(nltk.word_tokenize(para))
            for word, tag in tagged_text:
                if tag not in tags_all:
                    tags_all.append(tag)
    return tags_all


def f_pos(data, tags_all):
    tag_dict = {}
    for tag in tags_all:
        tag_dict[tag] = 0
    tag_count = {}
    for tag in tags_all:
        tag_count[tag] = []
    tag_count_body = {}
    for tag in tags_all:
        tag_count_body[tag] = []
    for title in data['title']:
        tagged_text = nltk.pos_tag(nltk.word_tokenize(title))
        for word, tag in tagged_text:
            tag_dict[tag] += 1
        for count,tag in zip(tag_dict.values(), tag_dict.keys()):
            tag_count[tag].append(count)
        tag_dict = {}
        for tag in tags_all:
            tag_dict[tag] = 0
    for body in data['usertext']:
        for para in body:
            tagged_text = nltk.pos_tag(nltk.word_tokenize(para))
            for word, tag in tagged_text:
                tag_dict[tag] += 1
        for count, tag in zip(tag_dict.values(), tag_dict.keys()):
            tag_count_body[tag].append(count)
        tag_dict = {}
        for tag in tags_all:
            tag_dict[tag] = 0
    return pd.concat((pd.DataFrame(tag_count, index=None), pd.DataFrame(tag_count_body, index=None)), axis=1)


def f_tfidf(data):
    X = []
    for t, b in zip(data['title'], data['usertext']):
        X.append(t + ' ' + b)
    count_vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_features=50)
    X_counts = count_vect.fit_transform(X)
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    return pd.DataFrame(X_tfidf.todense())


def f_topics(data, topic_num):
    def cleaning(article):
        punctuation = set(string.punctuation)
        lemmatize = WordNetLemmatizer()
        one = " ".join([i for i in article.lower().split() if i not in stopwords])
        two = "".join(i for i in one if i not in punctuation)
        three = " ".join(lemmatize.lemmatize(i) for i in two.lower().split())
        return three

    def pred_new(doc):
        one = cleaning(doc).split()
        two = dictionary.doc2bow(one)
        return two

    def load_title_body(data):
        text =[]
        for i in range(len(data["y"])):
            temp = str(data["title"][i])[2:-2]
            for j in data["usertext"][i]:
                temp = temp + ' ' + str(j)[2:-2]
            text.append(temp)
        return text

    stopwords = set(nltk.corpus.stopwords.words('english'))
    text_all = load_title_body(data)
    df = pd.DataFrame({'text': text_all}, index=None)
    text = df.applymap(cleaning)['text']
    text_list = []
    for t in text:
        temp = t.split()
        text_list.append([i for i in temp if i not in stopwords])

    dictionary = corpora.Dictionary(text_list)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_list]
    ldamodel = LdaModel(doc_term_matrix, num_topics=topic_num, id2word = dictionary, passes=50)
    probs = []
    for text in text_all:
        prob = ldamodel[(pred_new(text))]
        d = dict(prob)
        for i in range(topic_num):
            if i not in d.keys():
                d[i] = 0
        temp = []
        for i in range(topic_num):
            temp.append(d[i])
        probs.append(temp)
    return pd.DataFrame(probs, index=None)

def write_result_csv(data_dict, class_type):
    with open('../output/auc_{}.csv'.format(class_type), 'a') as f:  # replace this with your own output csv file
        field_names = ['date', 'model', 'auc', 'note']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerow(data_dict)


def find_threshold(fpr, tpr, threshold):
    rate = np.array(tpr) + np.array(fpr)
    return threshold[np.argmax(rate)]


def write_metrics_csv(data_dict, class_type):
    with open('../output/eval_{}.csv'.format(class_type), 'a') as f:  # replace this with your own output csv file
        field_names = ['date', 'model', 'accuracy', 'precision', 'recall', 'f-score', 'note', 'dataset']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerow(data_dict)


def predict_evaluate(y_test, y_pred, th, model_name, k_th, subreddit, class_type):
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


if __name__ == '__main__':
    class_type = 'imbalanced'  # considering whether the class is balanced or not
    subreddit = 'reddit'  # name of the dataset
    file_name = 'sample_data_1.xlsx'  # replace sample data with your input data
    df_data = pd.read_excel(file_name)

    # basic features
    print("Processing basic featues ...")
    df_basic = f_basic(df_data)

    # TF-IDF
    print("Processing TF-IDF features ...")
    df_tfidf = f_tfidf(df_data)

    # POS
    print("Processing POS features ...")
    tags_all = get_all_tags(df_data)
    df_pos = f_pos(df_data, tags_all)

    # Topics
    print("Processing Topics features ...")
    topic_num = 10
    df_topic = f_topics(df_data, topic_num)

    # # liwc features
    print("Processing LIWC features ...")
    df_liwc = f_liwc(class_type, subreddit)

    # concatenate all features
    df_features = pd.concat([df_basic, df_tfidf, df_pos, df_topic, df_liwc], axis=1)
    df_all = pd.concat([df_features, df_data['y']], axis=1)

    # under sampling
    num_sampling = 5
    for i in range(num_sampling):
        df_pos = df_all.loc[df_all['y'] == 1]
        df_neg = df_all.loc[df_all['y'] == 0]
        df_sample = pd.concat([df_pos, df_neg.sample(len(df_pos['y']))])
        df_sample = df_sample.fillna(0)
        # 10-fold cross validation
        X = df_sample[df_sample.columns[:-1]].as_matrix()
        y = df_sample['y'].as_matrix()
        num_fold = 10
        kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X):
            num_fold -= 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Logisitc Regression
            clf = LogisticRegression(penalty='l2', tol=1e-6)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:,1]
            fpr_lr, tpr_lr, th_lr = metrics.roc_curve(y_test, y_pred)
            roc_auc_lr = metrics.auc(fpr_lr, tpr_lr)
            result = {'date': datetime.date.today(), 'model': 'Logistic Regression', 'auc': roc_auc_lr, 'note': num_fold}
            write_result_csv(result, class_type)
            # o_lr = find_threshold(fpr_lr, tpr_lr, th_lr)
            o_lr = 0.5
            predict_evaluate(y_test, y_pred, o_lr, 'Logistic Regression', num_fold, subreddit, class_type)

            # Random Forest
            clf = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            fpr_rf, tpr_rf, th_rf = metrics.roc_curve(y_test, y_pred)
            roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
            result = {'date': datetime.date.today(), 'model': 'Random Foreset', 'auc': roc_auc_rf, 'note': num_fold}
            write_result_csv(result, class_type)
            # o_rf = find_threshold(fpr_rf, tpr_rf, th_rf)
            o_rf = 0.5
            predict_evaluate(y_test, y_pred, o_rf, 'Random Foreset', num_fold, subreddit, class_type)

            # GBDT
            clf = GradientBoostingClassifier(max_depth=8, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            fpr_gbdt, tpr_gbdt, th_gbdt = metrics.roc_curve(y_test, y_pred)
            roc_auc_gbdt = metrics.auc(fpr_gbdt, tpr_gbdt)
            result = {'date': datetime.date.today(), 'model': 'GBDT', 'auc': roc_auc_gbdt, 'note': num_fold}
            write_result_csv(result, class_type)
            # o_gbdt = find_threshold(fpr_gbdt, tpr_gbdt, th_gbdt)
            o_gbdt = 0.5
            predict_evaluate(y_test, y_pred, o_gbdt, 'GBDT', num_fold, subreddit, class_type)

            # XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, missing=-999)
            dtest = xgb.DMatrix(X_test, label=y_test, missing=-999)
            params = {'max_depth': 8, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic', 'nthread': -1}
            num_round = 10000
            watchlist = [(dtrain, 'train'), (dtest, 'test')]
            model = xgb.train(params, dtrain, num_round, watchlist, early_stopping_rounds=50, verbose_eval=10)
            y_pred = model.predict(dtest)
            fpr_xgb, tpr_xgb, th_xgb = metrics.roc_curve(y_test, y_pred)
            roc_auc_xgb = metrics.auc(fpr_xgb, tpr_xgb)
            result = {'date': datetime.date.today(), 'model': 'XGBoost', 'auc': roc_auc_xgb, 'note': num_fold}
            write_result_csv(result, class_type)
            # o_xgb = find_threshold(fpr_xgb, tpr_xgb, th_xgb)
            o_xgb = 0.5
            predict_evaluate(y_test, y_pred, o_xgb, 'XGBoost', num_fold, subreddit, class_type)

            # SVM
            print("\nSVM classification ...")
            clf2 = SVC(kernel='linear', C=1, probability=True)
            clf2.fit(X_train, y_train)
            y_pred = clf2.predict_proba(X_test)[:,1]
            fpr_svm, tpr_svm, th_svm = metrics.roc_curve(y_test, y_pred)
            roc_auc_svm = metrics.auc(fpr_svm, tpr_svm)
            result = {'date': datetime.date.today(), 'model': 'SVM', 'auc': roc_auc_svm, 'note': num_fold}
            write_result_csv(result, class_type)
            # o_svm = find_threshold(fpr_svm, tpr_svm, th_svm)
            o_svm = 0.5
            predict_evaluate(y_test, y_pred, o_svm, 'SVM', num_fold, subreddit, class_type)