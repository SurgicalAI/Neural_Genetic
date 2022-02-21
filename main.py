import numpy as np
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
#%matplotlib inline

import re
import string
import spacy

#misc:
import warnings
warnings.filterwarnings('ignore')
import tqdm as tqdm
import os

class Config:

    def __init__(self):

        #hyperparameters

        #paths
        self.train_file_path = '../input/feedback-prize-2021/train.csv'
        self.train_path = '../input/feedback-prize-2021/train'
        self.test_path = '../input/feedback-prize-2021/test'
        self.model_save_path = '.'

        # others

        conf = Config()

        train_df.head()

        train_df.predictionstring[0]
        train_df.info()

        #total number of unique text files

        len(train_df['id'].unique())

        #sample discourse text

        train_df['discourse_text'][0]

        #utility for printing the text
        def print_text(txt_id):
            with open(f'{conf.train_path}/{txt_id}.txt', 'r') as fp:
                text = fp.readlines()
                print(''.join(text))
         #corresponding text for above discourse text
        print_text('423A1CA112E2')
        #classes
        train_df['discourse_type'].unique().tolist()
        classes = ['Lead','Position','Evidence','Claim','Counterclaim','Rebuttal','Concluding Statement']
        discourse_type_num = train_df['discourse_type_num'].unique().tolist()
        discourse_type_num[0:5]

        # This function returns a list, where each index in the list correspond to a particular class (as in the list
        # classes declared above). Each entry in the returned list represents the maximum number of times a class has appeared in some
        # text(s).

        def get_list_of_max(classes, discourse_type_num):
            list_of_max = []

            for _class in classes:
                mx = 0
                for _type in discourse_type_num:
                    if _type[-1:].isdigit() and _type[-2:].isdigit():
                        if _class == _type[:-3]:
                            curr = _type[-2:]
                            mx = max(mx, int(curr))
                    else:
                        if _class == _type[:-2]:
                            curr = _type[-1:]
                            mx = max(mx, int(curr))

                list_of_max.append(mx)

            return list_of_max

        # get the maximum frequencies of all classes
        list_of_max = get_list_of_max(classes, discourse_type_num)
        plt.figure(figsize=(7, 5))

        ax = sns.barplot(x=list_of_max, y=classes)
        ax.set_xlabel('max frequency')
        ax.set_ylabel('classes')
        ax.set_title('Maximum number of times a class can be present in a text')

        # we can see from below graph that evidence and claim have appeared a maximum of 12 times in some text(s)
        plt.figure(figsize=(9, 7))
        sns.set_context('paper', font_scale=1.5)
        ax = sns.countplot(x='discourse_type', data=train_df)
        ax.set_xticklabels(classes, rotation=40, ha='right');
        dframe = train_df.drop_duplicates(subset=['id', 'discourse_type'], keep='first', inplace=False)
        dframe.head()
        len(dframe)
        plt.figure(figsize=(9, 7))
        sns.set_context('paper', font_scale=1.5)
        ax = sns.countplot(x='discourse_type', data=dframe)
        ax.set_xticklabels(classes, rotation=40, ha='right');
        df = train_df.copy()
        df['full_text'] = df['discourse_text'].groupby(df['id']).transform(lambda x: ' '.join(x))
        text_length = df['full_text'].drop_duplicates().apply(len)

        fig = plt.figure(figsize=(9, 7))

        ax = sns.distplot(text_length, kde=False, bins=100, color='r')
        ax.set_title('Distribution of Text Length')
        ax.set_xlabel("Text Length")
        ax.set_ylabel("Frequency")

        plt.show()
        word_count = df['full_text'].drop_duplicates().apply(lambda x: len(str(x).split()))

        fig = plt.figure(figsize=(9, 7))

        ax = sns.distplot(word_count, kde=False, bins=100, color='r')
        ax.set_title('Distribution of Word Count')
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")

        plt.show()

        df = train_df.copy()
        df['discourse_len'] = df['discourse_text'].apply(lambda x: len(x.split()))
        fig = plt.figure(figsize=(6, 5))

        ax = fig.add_axes([0, 0, 1, 1])
        ax = df.groupby('discourse_type')['discourse_len'].mean().sort_values().plot(kind="barh", color='lightseagreen')
        ax.set_title("Average number of words versus Discourse Type", fontsize=14)
        ax.set_xlabel("Average number of words", fontsize=12)
        ax.set_ylabel("")
        labels = ['Evidence', 'Concluding Statement', 'Lead', 'Rebuttal', 'Counterclaim', 'Position', 'Claim']
        feature = 'discourse_len'

        data = []
        for i in range(len(labels)):
            _data = df.loc[df['discourse_type'] == labels[i]][feature]
            data.append(_data)

        rows, cols = 4, 2
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 27))
        for idx, axis in enumerate(axes.reshape(-1)):
            if idx < len(labels):
                ax = sns.distplot(data[idx], ax=axis, color='lightseagreen')
                ax.set_title(labels[idx])
            if idx == len(labels):
                axis.axis('off')  # for turning off the axis of the very last plot
            plt.figure(figsize=(12, 5))
            ax = sns.kdeplot(data[0])
            ax = sns.kdeplot(data[1])
            ax = sns.kdeplot(data[2])
            ax = sns.kdeplot(data[3])
            ax = sns.kdeplot(data[4])
            ax = sns.kdeplot(data[5])
            ax = sns.kdeplot(data[6])
            ax.legend(labels)
            df['unique_words_count'] = train_df['discourse_text'].apply(lambda x: len(set(str(x).split())))
            fig = plt.figure(figsize=(6, 5))

            ax = fig.add_axes([0, 0, 1, 1])
            ax = df.groupby('discourse_type')['unique_words_count'].mean().sort_values().plot(kind="barh",
                                                                                              color='mediumturquoise')
            ax.set_title("Unique number of words versus Discourse Type", fontsize=14)
            ax.set_xlabel("Unique number of words", fontsize=12)
            ax.set_ylabel("")
            labels = ['Evidence', 'Concluding Statement', 'Lead', 'Rebuttal', 'Counterclaim', 'Position', 'Claim']
            feature = 'unique_words_count'

            data = []
            for i in range(len(labels)):
                _data = df.loc[df['discourse_type'] == labels[i]][feature]
                data.append(_data)

            rows, cols = 4, 2
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 27))
            for idx, axis in enumerate(axes.reshape(-1)):
                if idx < len(labels):
                    ax = sns.distplot(data[idx], ax=axis, color='teal')
                    ax.set_title(labels[idx])
                if idx == len(labels):
                    axis.axis('off')  # for turning off the axis of the very last plot
                plt.figure(figsize=(12, 5))
                ax = sns.kdeplot(data[0])
                ax = sns.kdeplot(data[1])
                ax = sns.kdeplot(data[2])
                ax = sns.kdeplot(data[3])
                ax = sns.kdeplot(data[4])
                ax = sns.kdeplot(data[5])
                ax = sns.kdeplot(data[6])
                ax.legend(labels)
                df['avg_word_length'] = train_df['discourse_text'].apply(
                    lambda x: np.mean([len(w) for w in str(x).split()]))
                fig = plt.figure(figsize=(6, 5))

                ax = fig.add_axes([0, 0, 1, 1])
                ax = df.groupby('discourse_type')['avg_word_length'].mean().sort_values().plot(kind="barh",
                                                                                               color='steelblue')
                ax.set_title("Average word length versus Discourse Type", fontsize=14)
                ax.set_xlabel("Average word length", fontsize=12)
                ax.set_ylabel("")
                labels = ['Evidence', 'Concluding Statement', 'Lead', 'Rebuttal', 'Counterclaim', 'Position', 'Claim']
                feature = 'avg_word_length'

                data = []
                for i in range(len(labels)):
                    _data = df.loc[df['discourse_type'] == labels[i]][feature]
                    data.append(_data)

                rows, cols = 4, 2
                fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 27))
                for idx, axis in enumerate(axes.reshape(-1)):
                    if idx < len(labels):
                        ax = sns.distplot(data[idx], ax=axis, color='steelblue')
                        ax.set_title(labels[idx])
                    if idx == len(labels):
                        axis.axis('off')  # for turning off the axis of the very last plot
                        plt.figure(figsize=(12, 5))
                        ax = sns.kdeplot(data[0])
                        ax = sns.kdeplot(data[1])
                        ax = sns.kdeplot(data[2])
                        ax = sns.kdeplot(data[3])
                        ax = sns.kdeplot(data[4])
                        ax = sns.kdeplot(data[5])
                        ax = sns.kdeplot(data[6])
                        ax.legend(labels)
                        data_df = train_df.groupby("discourse_type")[
                            ['discourse_end', 'discourse_start']].mean().reset_index().sort_values(
                            by='discourse_start',
                            ascending=False)

                        data_df.plot(x='discourse_type',
                                     kind='barh',
                                     stacked=False,
                                     title='Average start and end position absolute',
                                     figsize=(12, 4))

                        plt.show()

                        # creating temporary corpus for analysing punctuations and stopwords:
                        def temporary_corpus(target=None):
                            corpus = []

                            if target == None:
                                text_data = train_df['discourse_text'].str.split()
                            else:
                                text_data = train_df[train_df['discourse_type'] == target]['discourse_text'].str.split()

                            for text in text_data:
                                for char in text:
                                    corpus.append(char)
                            return corpus

                        plt.figure(figsize=(10, 5))
                        corpus = temporary_corpus()

                        dic = defaultdict(int)
                        special = string.punctuation
                        for token in corpus:
                            for character in token:
                                if character in special:
                                    dic[character] += 1

                        x, y = zip(*dic.items())
                        plt.bar(x, y, color='cadetblue')

                        import en_core_web_lg
                        nlp = en_core_web_lg.load()
                        corpus = temporary_corpus()

                        def get_frequent_stopwords(corpus):

                            dic = defaultdict(int)
                            for word in corpus:
                                if word in nlp.Defaults.stop_words:
                                    dic[word] += 1

                            # getting the top 20 most frequent stop words
                            top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:20]

                            return top, dic

                        top, dic = get_frequent_stopwords(corpus)
                    print(f'Total number of stop words in spacy: {len(nlp.Defaults.stop_words)}')
                    print(
                        f'Total number of stop words in the given text data which are part of the spacy\'s stop words list: {len(dic)}')
                    # top 20 most frequent stop words in our corpus:
                    plt.figure(figsize=(15, 4))
                    x, y = zip(*top)
                    plt.bar(x, y, color='goldenrod')

                    def get_count(text):
                        count = 0
                        for ch in text:
                            if ch in nlp.Defaults.stop_words:
                                count += 1
                        return count

                    df = train_df.copy()
                    df['stop_words_count'] = train_df['discourse_text'].apply(lambda x: get_count(x))
                    labels = ['Evidence', 'Concluding Statement', 'Lead', 'Rebuttal', 'Counterclaim', 'Position',
                              'Claim']

                    data = []
                    for i in range(len(labels)):
                        _data = df.loc[df['discourse_type'] == labels[i]]['stop_words_count']
                        data.append(_data)
                        plt.figure(figsize=(12, 5))
                        ax = sns.kdeplot(data[0])
                        ax = sns.kdeplot(data[1])
                        ax = sns.kdeplot(data[2])
                        ax = sns.kdeplot(data[3])
                        ax = sns.kdeplot(data[4])
                        ax = sns.kdeplot(data[5])
                        ax = sns.kdeplot(data[6])
                        ax.legend(labels)
                        from sklearn.feature_extraction.text import CountVectorizer
                        def get_n_grams(n_grams, top_n=10):

                            df_words = pd.DataFrame()

                            for dt in train_df['discourse_type'].unique():
                                df = train_df.query('discourse_type == @dt')
                                texts = df['discourse_text'].tolist()
                                vec = CountVectorizer(lowercase=True, stop_words='english', \
                                                      ngram_range=(n_grams, n_grams)).fit(texts)
                                bag_of_words = vec.transform(texts)
                                sum_words = bag_of_words.sum(axis=0)
                                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                                cvec_df = pd.DataFrame.from_records(words_freq, \
                                                                    columns=['words', 'counts']).sort_values(
                                    by="counts", ascending=False)
                                cvec_df.insert(0, "Discourse_type", dt)
                                cvec_df = cvec_df.iloc[:top_n, :]
                                df_words = df_words.append(cvec_df)

                            return df_words

                        unigrams = get_n_grams(n_grams=1, top_n=10)
                        unigrams.head()

                        def plot_ngram(df, type="unigrams"):

                            plt.figure(figsize=(15, 12))
                            plt.subplots_adjust(hspace=0.5)

                            for n, dt in enumerate(df.Discourse_type.unique()):
                                ax = plt.subplot(4, 2, n + 1)
                                ax.set_title(f"Most used {type} in {dt}")
                                data = df.query('Discourse_type == @dt')[['words', 'counts']].set_index(
                                    "words").sort_values(by="counts", ascending=True)
                                data.plot(ax=ax, kind='barh', color='steelblue')
                                plt.ylabel("")
                            plt.tight_layout()
                            plt.show()

                            # unigrams

                        plot_ngram(unigrams)

                        # bigrams
                        bigrams = get_n_grams(n_grams=2, top_n=10)
                        plot_ngram(bigrams, 'bigrams')

                        # trigrams
                        trigrams = get_n_grams(n_grams=3, top_n=10)
                        plot_ngram(trigrams, 'trigrams')

                        train_files = os.listdir(conf.train_path)
                        test_files = os.listdir(conf.test_path)

                        for file in range(len(train_files)):
                            train_files[file] = str(conf.train_path) + "/" + str(train_files[file])
                        for file in range(len(test_files)):
                            test_files[file] = str(conf.test_path) + "/" + str(test_files[file])

                            train_files[40]

                            train_files[40][35:-4]

                            r = 20
                            ents = []

                            for i, row in train_df[train_df['id'] == train_files[r][35:-4]].iterrows():
                                ents.append({
                                    'start': int(row['discourse_start']),
                                    'end': int(row['discourse_end']),
                                    'label': row['discourse_type']
                                })

                            with open(train_files[r], 'r') as file:
                                data = file.read()

                            doc2 = {
                                "text": data,
                                "ents": ents,
                            }

                            colors = {'Lead': 'turquoise',
                                      'Position': '#f9d5de',
                                      'Claim': '#adcfad',
                                      'Evidence': 'wheat',
                                      'Counterclaim': '#bdf2fa',
                                      'Concluding Statement': '#eea69e',
                                      'Rebuttal': '#d1f8f4'}

                            options = {"ents": train_df.discourse_type.unique().tolist(), "colors": colors}
                            spacy.displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)


