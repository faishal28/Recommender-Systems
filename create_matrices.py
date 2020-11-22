import pandas as pd
import numpy as np
import pickle as pickle
from scipy.sparse import csr_matrix
import scipy
import os
from time import time
start_time=time()

TEST_SIZE=0.2
SPARSE=True 
SHUFFLE=True

path='data/ratings.dat'
print('Data path:', path)
table=pd.read_table(path, sep='::', header=None, 
		names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

no_entries=table.shape[0]

if SHUFFLE:
	table = table.sample(frac=1).reset_index(drop=True)

movie_id_list=table['movieId'].unique()
user_id_list=table['userId'].unique()
no_users=len(user_id_list)
no_movies=len(movie_id_list)
print ('Overall dataset: total ratings=', no_entries)
print ('Overall dataset: total users=', len(user_id_list))
print ('Overall dataset: total movies=', len(movie_id_list))

movie_map={}
for ix, m_id in enumerate(movie_id_list):
	movie_map[m_id]=ix

user_map={}
for ix, m_id in enumerate(user_id_list):
	user_map[m_id]=ix

train_table=table.head(int((1-TEST_SIZE)*no_entries))
test_table=table.tail(int(TEST_SIZE*no_entries))

print ('Train set: total ratings=', train_table.shape[0])
print ('Train set: total users=', len(train_table['userId'].unique()))
print ('Train set: total movies=', len(train_table['movieId'].unique()))

print ('Test set: total ratings=', test_table.shape[0])
print ('Test set: total users=', len(test_table['userId'].unique()))
print ('Test set: total movies=', len(test_table['movieId'].unique()))

train = np.zeros([len(user_map), len(movie_map)])

print ('Creating matrices...')
create_start_time=time()
for idx,row in train_table.iterrows():
	train[user_map[row['userId']], movie_map[row['movieId']]]=row['rating']

print ('Time taken to create matrices:', time()-create_start_time)

print ('Train:', train.shape)
print ('Density:', 100.0*float(np.count_nonzero(train))/(no_users*no_movies))

sparse_start_time=time()
if SPARSE:
	train=scipy.sparse.csr_matrix(train)
print ('Time taken to convert to sparse:', time()-sparse_start_time)

if SPARSE:
	scipy.sparse.save_npz('temp_data/train.npz', train)
	if 'train.npy' in os.listdir('temp_data'):
		os.remove('temp_data/train.npy')

print ('train.npz saved to disk....')
with open('temp_data/movie_map.pkl', 'wb+') as f:
	pickle.dump(movie_map, f)
with open('temp_data/user_map.pkl', 'wb+') as f:
	pickle.dump(user_map, f)
with open('temp_data/test_table.pkl', 'wb+') as f:
	pickle.dump(test_table, f)
print ('movie_map.pkl and user_map.pkl saved to disk....')
print ('Script runtime:', time()-start_time)