import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
from pyspark import StorageLevel
from pyspark.sql.types import DateType

from pyspark.sql.window import Window as W
from pyspark.sql.functions import col, max as max_, min as min_
import pyspark.sql.functions as f
from pyspark.sql.functions import regexp_extract, col


import pickle 

import os
# Check whether the csv directory exists
if not os.path.exists('../data/csv/'):
    os.makedirs('../data/csv/')

#Pandas dataframe to save the filtered host and subreddit counts
OUT_FILE = '../data/csv/all_comments_filtered_host_sub_counts.csv

conf = SparkConf().setAll([('spark.executor.memory', '1500g'), 
                           ('spark.driver.memory','1500g'),
                           ('spark.ui.port','4046'),
                           ('spark.local.dir','/ada/tmp'),
                           ('spark.driver.maxResultSize','0')])
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
conf = SparkConf().setAll([('spark.executor.memory', '1500g'), 
                           ('spark.driver.memory','1500g'),
                           ('spark.ui.port','4048'),
                           ('spark.local.dir','/ada/tmp'),
                           ('spark.driver.maxResultSize','0')])
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

#Use the below code for scraping for comments with URLs

'''
def querying_data(WHERE, LIMIT=None, select = "*", parquet_location = "/ada/data/reddit/parquet/comments_2019.parquet", sample =False):
    df = spark.read.load(parquet_location)
    if sample == True:
        df = df.sample(fraction=0.05, withReplacement=False)
    df.createOrReplaceTempView("data")
    if LIMIT == None:
        data_queried = spark.sql("select {} from data WHERE {}".format(select, WHERE))
    else: 
        data_queried = spark.sql("select {} from data WHERE {} LIMIT {}".format(select, WHERE, LIMIT))
    return data_queried 

url_comments = querying_data(WHERE = "WHERE body LIKE '%http://%' OR body like '%https://%'")
url_comments.createOrReplaceTempView("url_comments")

regex = "((?:[a-z]+://|www\.|[^\s:=]+@www\.)([^/].*?[a-z0-9].*?)([a-z_\/0-9\-\#=&]|))(?=[\.,;\?\!]?(['«»\[\s\r\n]|$))"

result = url_comments.withColumn('url', regexp_extract(col('body'), regex, 2))
result.take(10)
result.write.parquet("data/2019_comments_url_method.parquet")
'''
#Load a parquet file with already parsed URLS for later processing and domain name extraction
resp = spark.read.load('/ada1/data/domainembed/url_comments.parquet')
resp.createOrReplaceTempView("response")

#Get host column
resp = resp.withColumn('host', regexp_extract(col('url'), '(https?:\/\/)?(www\.)?([A-Za-z_0-9.-]+).*', 3))
resp.createOrReplaceTempView("response")
resp = spark.sql("select author, subreddit, host,url,id,score from response")

#Filter out users that have posted less than 20 times
user_counts = resp.groupby(['author','host']).count()
user_counts_clean = user_counts.filter("count < 20")
grouped_host = user_counts_clean.groupBy('host').agg(f.sum("count").alias("sum_count"))
grouped_host = grouped_host.orderBy(col("sum_count").desc())
grouped_host_df = grouped_host.toPandas()

user_counts_clean = user_counts_clean.withColumn('key',f.concat_ws('_',user_counts_clean.author,user_counts_clean.host))
resp = resp.withColumn('key',f.concat_ws('_',resp.author,resp.host))
merged = resp.alias('a').join(user_counts_clean.alias('b'), resp.key == user_counts_clean.key).select("a.author", 
                                                                                "a.subreddit", 
                                                                                "a.host",
                                                                                "a.url",
                                                                                "a.id","a.key","a.score")
merged = merged.filter(merged.host.isin(list(filtered_hosts)))
#Remove all links to reddit
remove_reddit = merged.filter((merged.host != 'reddit.com') & (merged.host != 'old.reddit.com') & (merged.host != 'new.reddit.com'))
remove_reddit_df = remove_reddit.toPandas()
#Get only hosts that have been posted more than 200 times
many_shares = grouped_host_df[grouped_host_df.sum_count > 200]['host'].values
many_share_posts = remove_reddit_df.loc[remove_reddit_df.host.isin(many_shares)]
many_share_posts = pd.merge(many_share_posts,meta,on='host',how='inner')

many_share_posts.to_csv(OUT_FILE,index=False,header=True)