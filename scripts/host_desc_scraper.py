import pandas as pd
from bs4 import BeautifulSoup
import regex as re
import requests
import argparse
from queue import Queue
import pickle
import time
import threading
from tqdm import tqdm
#session = HTMLSession()
lock = threading.Lock()

def update_last_vid(df,host,desc):
	lock.acquire() # thread blocks at this line until it can obtain lock
	# in this section, only one thread can be present at a time.
	df.loc[df['host'] == host, 'desc'] = desc
	lock.release()

def get_desc(host):
	try:
		page_source = requests.get("https://" + host,timeout=5).text
		soup = BeautifulSoup(page_source,'html.parser')
		name_results = soup.find_all('meta',{'name': re.compile(r'desc|Desc')})
		desc = name_results[0]["content"]
		return desc
	except:
		#print('Error getting host: {}'.format(host))
		return ''

def worker_loop(worker_id,queue,df,pbar):
	while not queue.empty():
		host = queue.get()
		desc = get_desc(host)
		update_last_vid(df,host,desc)
		queue.task_done()
		pbar.update(1)
        
def main():
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument("--from_id", type = int, default = 0)
	parser.add_argument("--to_id", type = int, default = 10000)
	parser.add_argument("--num_workers", type=int,default=5)
	args = parser.parse_args()

	META_FILE = '../../embeddings/all_filtered_meta.tsv'
	BASE = '../../data/csv/'

	meta = pd.read_csv(META_FILE,sep='\t')

	host_list = meta.iloc[args.from_id:args.to_id]['host'].values
	#host_list = ['twitter.com','imgur.com','youtube.com','cnn.com']

	#host_list = ['localhost']*50
	#Initialize queue
	queue = Queue()
	for host in host_list:
		queue.put(host)
	#print('Queue is full')
	pbar = tqdm(total=args.to_id-args.from_id)
	for i in range(args.num_workers):
		t = threading.Thread(target=worker_loop,args=(i,queue,meta,pbar))
		t.start()

	#print('*** Main thread waiting')
	queue.join()
	#print ('*** Done getting descriptions')
	end = time.time()
	print('time took: {}'.format(end-start))
	output_file = meta.iloc[args.from_id:args.to_id]
	#with open('desc_list_{}_{}.pickle'.format(args.from_id,args.to_id),'wb') as file:
		#pickle.dump(desc_list,file)
	output_file.to_csv(BASE + 'desc_list_{}_{}.csv'.format(args.from_id,args.to_id),index=False,header=True)

if __name__ == "__main__":
	main()
