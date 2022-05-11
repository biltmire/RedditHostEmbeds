# Social Map of the Internet
## scripts
1. In order to create the embeddings first you have to run get_url_comments in order to generate the file of subreddit/host counts.

2. To create the host embeddings run domain_embed_creation.py which takes in the output file generated in the previous step in order to weight the reddit embeddings.

3. To get the top PMI words for each cluster run get_top_PMI_words.py. This will create a pickle file with the cluster word frequency dictionaries.

4. To generate cluster summaries using the pre-existing labels and cluster word frequency lists generated in the last step, run the generate_cluster_summary.py script.

5. To scrape the top 10k websites to get the descriptions run host_desc_scraper.py.

6. To generate BERT embeddings based on these descriptions run transformer_vec_grabber.py.

## notebooks
Domains - Dimension scoring.ipynb: To generate the plots scoring the host vectors on different social dimensions and visualize the results.
