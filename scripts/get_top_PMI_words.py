import pandas as pd
import numpy as np
import pickle

from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
from pyspark import StorageLevel
from pyspark.sql.types import DateType

from pyspark.sql.window import Window as W
from pyspark.sql.functions import col, max as max_, min as min_
import pyspark.sql.functions as f
from pyspark.sql.functions import regexp_extract, col

from sklearn.feature_extraction.text import CountVectorizer
import regex as re

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tqdm

alphabet = 'abcdefghijklmnopqrstuvwxyz'
labels = ['Europe','Videogame Reviews','British Japanese Learners','Hockey','US Politics','Anime 1','US Moving Resources','Tech Servers','Outdoors','XBOX','PC Games','Videogames Crowdfunding','Videogames Blizzard','Marijuana','Smartphones','Web development','Photo & Video sharing','PC Building','Videogame Twitch Streaming','Self Help Books','3D Printing DIY','Fashion Men','Fantasy Sports','Star Wars DC Marvel','Australia','Web Comics Movie Reviews??','Science','Finance','Racing','Fashion Female','Soccer','Fitness','Pornography','Leauge of Legends','GIF Creation','Music Production','Card Games','Religious','UK Politics','Canada','Font Sharing','Cooking','Cyrpto currency','Bicycles','Videogame Military Sim','Firearms','Investing','India','Board Games','Cartoon Fan Fiction']

def remove_urls(token):
    regex = r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&=]*)"
    regex2 = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)"
    return re.sub("|".join([regex,regex2]), '',token)

def latin(word):
    for char in word:
        if char not in alphabet:
            return False
    return True

conf = SparkConf().setAll([('spark.executor.memory', '1500g'),
                           ('spark.driver.memory','1500g'),
                           ('spark.ui.port','4048'),
                           ('spark.local.dir','/ada/tmp'),
                           ('spark.driver.maxResultSize','0')])
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

#Load host vector metadata
meta = pd.read_csv('../embeddings/all_filtered_meta.tsv',sep='\t')

#Load Reddit User Comments parquet that has body field (or the field with the actual comment text)
resp = spark.read.load('../data/parquet/2019_comments_url_method.parquet')
resp.createOrReplaceTempView("response")
resp = resp.withColumn('host', regexp_extract(col('url'), '(https?:\/\/)?(www\.)?([A-Za-z_0-9.-]+).*', 3))
embed_domains = resp.filter(resp.host.isin(list(meta.host.values)))

#Dic of word frequencies in each cluster
cluster_word_freqs = {}

#Create the word cluster frequency lists
for label in meta.labels.unique():
    print('starting with label: {}'.format(label))
    label_domains = list(meta.loc[meta.labels == label]['host'].values)
    domains = embed_domains.filter(embed_domains.host.isin(label_domains))
    body_vals = domains.select("body").rdd.flatMap(lambda x: x).collect()
    clean_vals = [remove_urls(val) for val in body_vals]
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(clean_vals)
    sum_words = X.sum(axis=0)
    words_freq = [(word,sum_words[0,idx]) for word,idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq,key=lambda x:x[1],reverse=True)
    cluster_word_freqs[label] = words_freq
    print('done with label: {}'.format(label))

#Code to generate the cleaned super_dictionary from the cluster_word_freqs
super_dict = {}
keys = cluster_word_freqs.keys()

for key in keys:
    super_dict[key] = dict(cluster_word_freqs[key])

#clean dicts
for key in keys:
    count = 0
    dic = super_dict[key]
    for word in list(dic.keys()):
        if not (latin(word) and d.check(word)):
            del dic[word]
            count += 1
    print('Dictionary: {} lost {} words'.format(key,count))

#Create a dictionary of the total number of times a word in the super_dict appears in the corpus
total_count_dict = {}
def num_words(word):
    word_counts = []
    for key in keys:
        if word in super_dict[key]:
            word_counts.append(super_dict[key][word])
    return(sum(word_counts))
total_count_dict = {}
for key in super_dict.keys():
    dic = super_dict[key]
    for word in dic.keys():
        if word not in total_count_dict:
            total_count_dict[word] = num_words(word)

'''
If w is a word and c is a cluster and N is the size of the corpus:
P(w,c) = n(w in c)/N,
P(w) = n(w)/N
p(c) = Σ word counts for words in c/N

PMI(w,c) = log(N*n(w in c)/(n(w)*Σ word counts for words in c))
'''
def c_sum(wrd_dic):
    return sum([wrd_dic[word] for word in wrd_dic.keys()])
N = sum([c_sum(super_dict[key]) for key in keys])

pmi_dict = {}
from tqdm.notebook import tqdm
for key in tqdm(cluster_word_freqs.keys()):
    cluster_list = super_dict[key]
    n_c = c_sum(cluster_list)
    pmi_vals = []
    for word in cluster_list:
        n_w_c = cluster_list[word]
        n_w = total_count_dict[word]
        if n_w >= 500:
            pmi = np.log(N*n_w_c/(n_c*n_w))
            pmi_vals.append((word,pmi))
    pmi_dict[key] = sorted(pmi_vals,key=lambda x:x[1],reverse=True)

with open('../data/pickle/filtered_hosts_2019_pmi_dict.pickle', 'wb') as handle:
    pickle.dump(pmi_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

for label in tqdm(sorted(meta.labels.unique())):
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=dict(pmi_dict[label]))
    plt.figure(figsize=(11,9),facecolor='white')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Axis label: {}".format(labels[label]))
    plt.savefig('../plots/wordclouds/filtered_pmi_cluster_{}_wordcloud.png'.format(labels[label]))
