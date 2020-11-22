import numpy as np
import recsys_utils
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
from time import time
from copy import deepcopy as dc
import evaluation
from tqdm import tqdm

def subtract_mean(matrix_):
	matrix = dc(matrix_)
	c = Counter(matrix.nonzero()[0])
	means_matrix = matrix.sum(axis=1)
	means_matrix = np.reshape(means_matrix, [means_matrix.shape[0], 1])
	for i in range(means_matrix.shape[0]):
				if i in list(c.keys()):
					means_matrix[i,0] = means_matrix[i, 0]/float(c[i])
				else:
					means_matrix[i,0] = 0
	mask = matrix!=0
	nonzero_vals = np.array(np.nonzero(matrix))
	nonzero_vals = list(zip(nonzero_vals[0], nonzero_vals[1]))
	for val in nonzero_vals:
		matrix[val[0], val[1]] -= means_matrix[val[0]]
	return matrix

def predict(matrix, dist_matrix, testing, user_mappings, movie_mappings, n=10, mode='user'):
	pred=[]
	if mode=='user':
		# testing cases
		for idx,row in testing.iterrows():
			dist=np.reshape(dist_matrix[:, user_mappings[row['userId']]], [len(dist_matrix),1])
			usr_ratings=matrix[:, movie_mappings[row['movieId']]].todense()
			temp_rating_dist=list(zip(dist.tolist(), usr_ratings.tolist()))
			temp_rating_dist.sort(reverse=True)
			temp_rating_dist=temp_rating_dist[1:]
			rating=0
			c=1
			den=0
			for i in range(len(temp_rating_dist)):
				if c>=n:
					break
				elif temp_rating_dist[i][1][0]!=0:
					rating+=temp_rating_dist[i][1][0]*temp_rating_dist[i][0][0]
					den+=temp_rating_dist[i][0][0]
				c+=1
			if den==0:
				den=1
			rating=rating/den
			pred.append(rating)


	else:
		for idx,row in testing.iterrows():
			dist=np.reshape(dist_matrix[:, movie_mappings[row['movieId']]], [len(dist_matrix),1])
			movie_ratings=matrix[:, user_mappings[row['userId']]].todense()
			temp_rating_dist=list(zip(dist.tolist(), movie_ratings.tolist()))
			temp_rating_dist.sort(reverse=True)
			temp_rating_dist=temp_rating_dist[1:]
			rating=0
			c=1
			den=0
			for i in range(len(temp_rating_dist)):
				if c>=n:
					break
				elif temp_rating_dist[i][1][0]!=0:
					rating+=temp_rating_dist[i][1][0]*temp_rating_dist[i][0][0]
					den+=temp_rating_dist[i][0][0]
				c+=1
			if den==0:
				den=1
			rating=rating/den
			pred.append(rating)
	return np.array(pred)

if __name__=='__main__':
	training=recsys_utils.read_train(sparse=True)
	testing=recsys_utils.read_test_table().head(10000)
	truth=testing['rating'].head(10000).as_matrix()
	user_mappings=recsys_utils.read_user_map()
	movie_mappings=recsys_utils.read_movie_map()

	user_means=np.squeeze(np.sum(np.array(training.todense()), axis=1))
	user_means=np.divide(user_means, (np.array(training.todense())!=0).sum(1))
	print('collaborative filtering for User-User:')
	start_time_user=time()
	user_dist=1-pairwise_distances(subtract_mean(training.astype('float32')), metric='cosine')
	print('distance calculation time:', time()-start_time_user)
	predictions=predict(training, user_dist, testing, user_mappings, movie_mappings, 10)
	print('Time for User-User:', time()- start_time_user)
	print('RMSE:', evaluation.RMSE(predictions, truth))
	print('spearman rank correlation:', evaluation.spearman_rank_correlation(predictions, truth))
	print('top k precision:', evaluation.top_k_precision(predictions, testing, user_means, user_mappings, k=5))
	print('Total time:', time()-start_time_user)

	# Item-item collaborative filtering
	it_means=np.squeeze(np.sum(np.array(training.T.todense()), axis=1))
	it_means=np.divide(it_means, (np.array(training.T.todense())!=0).sum(1))
	print('collaborative filtering for Item-Item:')
	start_time_item=time()
	item_dist=1-pairwise_distances(subtract_mean(training.T.astype('float32')), metric='cosine')
	print('Time taken to calculate distances:', time()-start_time_item)
	predictions=predict(training.T, item_dist, testing, user_mappings, movie_mappings, 10, 'item')
	print('Time for Item-Item:', time()- start_time_item)
	print('RMSE:', evaluation.RMSE(predictions, truth))
	print('spearman rank correlation:', evaluation.spearman_rank_correlation(predictions, truth))
	print('top k precision:', evaluation.top_k_precision(predictions, testing, it_means, movie_mappings, k=5, user_=False))
	print('Total time:', time()-start_time_item)