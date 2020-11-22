import pandas as pd
import numpy as np
from time import time
import recsys_utils
import os
from copy import deepcopy as dc
import evaluation


def energy_calculation(v, retain_energy):
	if v.ndim==2:
		v=np.squeeze(v)
	elif retain_energy==0:
		return -1
	print(v[0:10])
	total_energy=np.sum(v)
	required_energy=retain_energy*total_energy/(100.0)
	index=np.argmin(v.cumsum() <= required_energy)+1
	return index

def SVD(matrix, retain_energy=90, save_factorized=False):
	vals, vs=np.linalg.eig(np.dot(matrix, matrix.T))
	vals=np.absolute(np.real(vals))
	if retain_energy==100:
		no_ev=np.linalg.mat_rank(np.dot(matrix, matrix.T))
	else:
		no_ev=energy_calculation(np.sort(vals)[::-1], retain_energy)
	print('No of ev retained:', no_ev)
	indices=np.argsort(vals)[::-1][0:no_ev]
	U=np.real(vs[:, indices])

	diag_vals=dc(np.reshape(np.sqrt(np.sort(vals)[::-1])[0:no_ev], [no_ev]))

	#  sigma calculation
	sigma=np.zeros([no_ev, no_ev])
	np.fill_diagonal(sigma, diag_vals)

	# V calculation
	V=np.zeros([matrix.shape[1], no_ev])
	for i in range(no_ev):
		scaling_factor=(1/diag_vals[i])
		V[:, i]= scaling_factor*np.reshape(np.dot(matrix.T, np.reshape(U[:, i], [U.shape[0], 1])), [matrix.shape[1]])
	V_t=V.T

	if save_factorized:
		np.save('temp_data/U', U)
		np.save('temp_data/V_t', V_t)
		np.save('temp_data/sigma', sigma)
		print('matrices saved!')

	return U, V_t, sigma

if __name__=='__main__':
	# Read data
	train=np.array(recsys_utils.read_train())
	test=recsys_utils.read_test_table()
	truth=test['rating'].as_matrix()
	user_map=recsys_utils.read_user_map()
	movie_map=recsys_utils.read_movie_map()

	start_time=time()

	# Subtracting mean of data from train set
	user_means=np.squeeze(np.sum(train, axis=1))
	user_means=np.divide(user_means, (train!=0).sum(1))
	for i in range(train.shape[0]):
		train[i, :][train[i, :]!=0]-=user_means[i]

	# Decomposition and Reconstruction of SVD
	U, V_t, sigma=SVD(train, retain_energy=90, save_factorized=True)
	reconstructed=np.dot(np.dot(U, sigma), V_t)

	# Get predicted
	pred_matrix=train+np.reshape(user_means, [len(user_means), 1])
	ro=[user_map[x] for x in test['userId']]
	co=[movie_map[x] for x in test['movieId']]
	predicted=pred_matrix[ro, co]
	total_time_svd=time()-start_time
	print('RMSE:', evaluation.RMSE(np.array(predicted), truth))
	print('spearman_rank_correlation', evaluation.spearman_rank_correlation(np.array(predicted), truth))
	print('Top k Precision(k=5):', evaluation.top_k_precision(predicted, 
		test, user_means, user_map, 5))
	print('Total SVD time:', total_time_svd)