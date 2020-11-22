import numpy as np

def RMSE(pred, truth):
	return np.sqrt(np.sum(np.square(pred-truth)/float(pred.shape[0])))

def RMSE_mat(matA, matB):
	return np.sqrt(np.sum(np.square(matA-matB))/(matA.shape[0]*matA.shape[1]))

def top_k_precision(pred, test, means_, map_, k=5, user_=True):
	K=k
	precision_list=[]
	print('test shape', test.shape, 'pred shape', pred.shape)
	test['prediction']=pred

	if user_==True:
		unique_values=test['userId'].unique()
	else:
		unique_values=test['movieId'].unique()

	for val in unique_values:
		THRESHOLD=means_[map_[val]]
		if user_==True:
			temp_df=test[test['userId']==val].copy(deep=True)
		else:
			temp_df=test[test['movieId']==val].copy(deep=True)
		temp_df.sort_values('prediction', inplace=True, ascending=False)
		temp_df=temp_df.head(K)
		temp_df['rating']=temp_df['rating']>=THRESHOLD
		temp_df['prediction']=temp_df['prediction']>=THRESHOLD
		no_equals = temp_df[temp_df["rating"] == temp_df["prediction"]].shape[0]
		temp_precision=no_equals/float(temp_df.shape[0])
		# print no_equals, temp_precision
		precision_list.append(temp_precision)
	return np.mean(np.array(precision_list))

def spearman_rank_correlation(pred, truth):
	d=np.sum(np.square(pred-truth))
	n=len(pred)
	rho=1-6.0*d/(n*(n*n-1))
	return rho

if __name__=='__main__':
	shp=[100, 100]
	a=np.random.randint(1, 6, shp)
	b=np.random.randint(1, 6, shp)
	print(RMSE_mat(a,b))
	print(spearman_rank_correlation(a,b))