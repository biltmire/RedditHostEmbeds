import pandas as pd
import numpy as np
import tqdm

def return_vector(host,vectors,meta):
    counts = top_host_sub_counts.loc[top_host_sub_counts.host == host][['subreddit','count']].values
    sub_list = counts[:,0]
    count_list = counts[:,1]
    index_list = meta.loc[meta.community.isin(sub_list)].reset_index().set_index('community')[['index']]
    index_list = index_list.reindex(sub_list)['index'].values
    #Multiply each vector for a subreddit by its appear count for that host and sum all the resulting values
    return_vec = vectors.iloc[index_list].mul(count_list,axis=0).sum(axis=0)
    #Return the summed vector divided by the total number of post counts for this website
    return return_vec/sum(count_list)

#Load the vectors and metadata from the subreddit vectors
vectors = pd.read_csv('../embeddings/reddit_vectors.tsv',sep='\t',header=None)
meta = pd.read_csv('../embeddings/reddit_metadata.tsv',sep='\t')

#Load host and subreddit counts
filtered_sub_counts = pd.read_csv('../data/csv/all_comments_filtered_host_sub_counts.csv')

#Get top 10k most posted hosts
host_counts = filtered_sub_counts.groupby('host')['count'].sum()
top_10k = host_counts.sort_values(ascending=False)[:10000].index.values
top_host_sub_counts = filtered_sub_counts.loc[filtered_sub_counts.host.isin(top_10k)]

vector_list = pd.DataFrame()
for host in tqdm(top_10k):
    vector = return_vector(host,vectors,meta)
    vector_list = vector_list.append(vector,ignore_index=True)

#Normalize embeddings
print('Saving vectors and metadata')
domain_vectors = vector_list.divide(np.linalg.norm(vector_list, axis=1), axis=0)
domain_vectors.to_csv('../embeddings/all_filtered_vecs.tsv',sep='\t',index=False,header=False)

domain_meta = pd.DataFrame(top_10k,columns=['host'])
domain_meta.to_csv('../embeddings/all_filtered_meta.tsv',sep='\t',index=False,header=True)
