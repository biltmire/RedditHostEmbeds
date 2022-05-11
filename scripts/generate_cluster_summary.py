import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors.nearest_centroid import NearestCentroid
from tqdm import tqdm
from os import chdir
import pickle
#Get the top 10 euclidean distances from each cluster
from scipy.spatial import distance

def pmi(dff,x,y):
    df = dff.copy()
    xy = df.groupby([x,y])['count'].sum()
    xy = xy.to_frame().reset_index()
    x_count = df.groupby(x)['count'].sum().to_frame()
    x_count.rename(columns={'count':x + '_count'},inplace=True)
    y_count = df.groupby(y)['count'].sum().to_frame()
    y_count.rename(columns={'count':y + '_count'},inplace=True)
    xy = pd.merge(xy,x_count,on=x)
    xy = pd.merge(xy,y_count,on=y)
    N = sum(xy['count'])
    xy['pmi'] = np.log(N * xy['count'] / (xy[x + '_count'] * xy[y + '_count']))
    if x == 'subreddit' or y == 'subreddit':
        xy = xy[~xy['subreddit'].str.contains('u_')]
    xy= xy.sort_values(by='pmi', ascending = False)
    return xy

def get_top_10_dists(cosine=False,all_vecs=False,num=10):
    '''
    Get top 10 Euclidean distance(cosine=False) or cosine similarity on all vectors(all_vecs=True) or just from the
    vectors of each label excusively
    '''
    top_10_center_dists = []
    for i in tqdm(list(range(N_CLUSTERS))):
        centroid = centroids[i]
        if not all_vecs:
            label_indices = domain_meta[domain_meta.labels == i].index
        else:
            label_indices = domain_meta.index
        label_vecs = domain_vectors.iloc[label_indices]
        if not cosine:
            distances = np.linalg.norm(label_vecs - centroid,axis = 1)
        else:
            distances = [1 - distance.cosine(np.array(val[1]),centroid) for val in label_vecs.iterrows()]
        top_10_idx = np.argsort(distances)[-num:]
        top_distances = [distances[j] for j in top_10_idx]
        top_10 = domain_meta.iloc[label_indices[top_10_idx]][['host','labels']].values
        final_list = [list(top_10[j]) + [top_distances[j]] for j in range(10)]
        top_10_center_dists.append(final_list[::-1])
    return top_10_center_dists

ROOT = "~/domain_embeddings/"
chdir(ROOT)

VECS_FILE = "embeddings/all_filtered_vecs.tsv"
META_FILE = "embeddings/all_filtered_meta.tsv"
COUNT_FILE = "data/csv/all_comments_filtered_host_sub_counts.csv"
WORD_PMI_FILE = "data/pickle/filtered_hosts_2019_pmi_dict.pickle"
N_CLUSTERS = 50
#Filename to save the cluster experiments file
filename = 'data/txt/' + 'all_filtered_{}clusters.txt'.format(N_CLUSTERS)
#Flags
HAVE_LABELS = True
RESAVE_META = True
#READ DATA
print('Loading vecs and meta')
domain_vectors = pd.read_csv(VECS_FILE,sep='\t',header=None)
domain_meta = pd.read_csv(META_FILE,sep='\t')
print('Done loading vecs and meta')
#Read count file which is formatted like:
'''
host,subreddit,count
youtube.com,AskYoutube,234343
..., ..., ...
..., ..., ...
..., ..., ...
..., ..., ...
..., ..., ...
'''
print('Loading count data')
filtered_sub_counts = pd.read_csv(COUNT_FILE)
print('Done loading count data')
model = AgglomerativeClustering(distance_threshold=None,linkage='ward', n_clusters=N_CLUSTERS)
model = model.fit(domain_vectors)
domain_meta['labels'] = model.labels_
filtered_sub_counts = pd.merge(filtered_sub_counts,domain_meta,on='host',how='inner')

print("Generating PMI vals")
pmi_frame = pmi(filtered_sub_counts,'labels','subreddit')
print("Done generating PMI vals")
#Top PMI with post threshold
post_thresh = 1000
top_pmi_post_thresh = pmi_frame.sort_values(by='pmi',ascending=False).loc[pmi_frame.subreddit_count > post_thresh].groupby('labels').head(10)
#Top post PMI thresh
pmi_thresh = 3
top_post_pmi_thresh = pmi_frame.sort_values(by='subreddit_count',ascending=False).loc[pmi_frame.pmi > pmi_thresh].groupby('labels').head(10)

#Get the most posted hosts in each cluster
top_cluster_hosts = filtered_sub_counts.groupby(['labels','host'])['count'].sum().reset_index()
top_cluster_hosts = top_cluster_hosts.sort_values(by='count',ascending=False).groupby('labels').head(10)

#Get centroids of each cluster
clf = NearestCentroid()
clf.fit(domain_vectors,domain_meta['labels'])
centroids = clf.centroids_

print('Getting closest hosts to centroids')
top_10_sin_dists = get_top_10_dists(True,True)

#Get number of hosts in each cluster
cluster_sizes = domain_meta.labels.value_counts()

#If the clusters have been manually labelled add these to the output file and create a clustering file for
#use with the dimension scoring
if HAVE_LABELS:
    cluster_file = ROOT + 'data/csv/clustering/filtered_{}_cluster_file.csv'.format(N_CLUSTERS)
    #Put manual labels here
    labels = ['Europe','Videogame Reviews','British Japanese Learners','Hockey','US Politics','Anime 1','US Moving Resources','Tech Servers','Outdoors','XBOX','PC Games','Videogames Crowdfunding','Videogames Blizzard','Marijuana','Smartphones','Web development','Photo & Video sharing','PC Building','Videogame Twitch Streaming','Self Help Books','3D Printing DIY','Fashion Men','Fantasy Sports','Star Wars DC Marvel','Australia','Web Comics Movie Reviews??','Science','Finance','Racing','Fashion Female','Soccer','Fitness','Pornography','Leauge of Legends','GIF Creation','Music Production','Card Games','Religious','UK Politics','Canada','Font Sharing','Cooking','Cyrpto currency','Bicycles','Videogame Military Sim','Firearms','Investing','India','Board Games','Cartoon Fan Fiction']
    assert len(labels) == N_CLUSTERS
    domain_meta['cluster_name'] = [labels[i] for i in domain_meta['labels']]
    print('Saving cluster File')
    domain_meta[['host','labels','cluster_name']].rename(columns={'labels':'cluster_id'}).to_csv(cluster_file,index=False,header=True)

if RESAVE_META:
    print('Resaving META')
    domain_meta.to_csv(META_FILE,sep='\t',header=True,index=False)

#Create a list of top 10 words for each cluster if PMI file is not empty
if WORD_PMI_FILE:
    #Load word frequency dictionaries
    print('Loading WORD PMI File')
    with open(WORD_PMI_FILE,'rb') as file:
        pmi_dict = pickle.load(file)
    #top_10_words_list = [pmi_dict[label][:10] for label in list(range(N_CLUSTERS))]
    word_pmi_count_thresh_list = []
    word_count_pmi_thresh_list = []
    count_thresh = 10000
    pmi_thresh = 3
    print('Sorting words by PMI and Count with PMI/Count threshold')
    for key in list(range(N_CLUSTERS)):
        word_df = pd.DataFrame(pmi_dict[key],columns=['word','pmi','count'])
        top_pmi_count_thresh = word_df.loc[word_df['count'] > count_thresh].sort_values(by='pmi',ascending=False).head(10)
        word_pmi_count_thresh_list.append(top_pmi_count_thresh[['word','pmi','count']].values)
        #Top post PMI thresh
        top_count_pmi_thresh = word_df.sort_values(by='count',ascending=False).loc[word_df.pmi > pmi_thresh].head(10)
        word_count_pmi_thresh_list.append(top_count_pmi_thresh[['word','pmi','count']].values)

#Write to file (it looks ugly I know)
with open(filename,'w') as f:
    for i in range(N_CLUSTERS):
        top_sub_pmi = top_pmi_post_thresh.loc[top_pmi_post_thresh.labels == i][['subreddit','pmi','subreddit_count']].values
        top_sub_post = top_post_pmi_thresh.loc[top_post_pmi_thresh.labels == i][['subreddit','pmi','subreddit_count']].values
        top_10_hosts = top_cluster_hosts.loc[top_cluster_hosts.labels == i][['host','count']].values
        top_10_closest = top_10_sin_dists[i]
        if WORD_PMI_FILE:
            top_word_pmi = word_pmi_count_thresh_list[i]
            top_word_count = word_count_pmi_thresh_list[i]
            final_word_pmi_string = 'Top 10 PMI words with count in cluster threshold:\n'
            final_word_count_string = 'Top 10 Count words with PMI in cluster threshold:\n'
        else:
            final_word_pmi_string = ''
            final_word_count_string = ''
        if HAVE_LABELS:
            label = labels[i]
        else:
            label = 'N/A'
        cluster_string = '----------Cluster: {} (Desc: {}), Size: {}----------\n'.format(i,label,cluster_sizes.loc[i])
        final_pmi_str = 'Top PMI with post threshold:\n'
        final_sub_string = 'Top Subreddits with PMI threshold:\n'
        final_posted_string = 'Most posted sites:\n'
        final_closest_string = 'Hosts with closest cosine similarity to centroid:\n'
        for j in range(10):
            pmi_dat = top_sub_pmi[j]
            try:
                sub_dat = top_sub_post[j]
            except:
                sub_dat = ['N/A','N/A','N/A']
            final_pmi_str += '{} - Subreddit: {}, PMI: {}, Subreddit count: {}\n'.format(j+1,pmi_dat[0],pmi_dat[1],pmi_dat[2])
            final_sub_string += '{} - Subreddit: {}, Subreddit count: {}, PMI: {}\n'.format(j+1,sub_dat[0],sub_dat[2],sub_dat[1])
            final_posted_string += '{} - Host: {}, Post Count: {}\n'.format(j+1,top_10_hosts[j][0],top_10_hosts[j][1])
            final_closest_string += '{} - Host: {}, Label: {}, Distance: {}\n'.format(j+1,top_10_closest[j][0],top_10_closest[j][1],top_10_closest[j][2])
            if WORD_PMI_FILE:
                try:
                    word_pmi_dat = top_word_pmi[j]
                except:
                    word_pmi_dat = ['N/A','N/A','N/A']
                try:
                    word_count_dat = top_word_count[j]
                except:
                    word_count_dat = ['N/A','N/A','N/A']
                final_word_pmi_string += '{} - Word: {}, PMI to cluster: {}, Count in cluster: {}\n'.format(j+1,word_pmi_dat[0],word_pmi_dat[1],word_pmi_dat[2])
                final_word_count_string += '{} - Word: {}, Count in cluster: {}, PMI in cluster: {}\n'.format(j+1,word_count_dat[0],word_count_dat[2],word_count_dat[1])
        #Append to final cluster string
        cluster_string += final_pmi_str + final_sub_string + final_posted_string + final_closest_string + final_word_pmi_string + final_word_count_string + '\n'
        f.write(cluster_string)
f.close()
