
import numpy as np
import recsys_utils
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
from time import time
from copy import deepcopy as dc
import evaluation


training=recsys_utils.read_train(sparse = True)
testing=recsys_utils.read_test_table()
truth=testing['rating'].as_matrix()
user_map=recsys_utils.read_user_map()
movie_map=recsys_utils.read_movie_map()


def sub_mean(matrix_,type='user'):
    matrix=dc(matrix_)
    c=Counter(matrix.nonzero()[0])
    means_matrix=matrix.sum(axis=1)
    means_matrix=np.reshape(means_matrix, [means_matrix.shape[0], 1])
    for i in range(means_matrix.shape[0]):
            if i in list(c.keys()):
                means_matrix[i,0]=means_matrix[i, 0]/float(c[i])
            else:
                means_matrix[i,0]=0
   
    mask=matrix!=0
    nz_val=np.array(np.nonzero(matrix))
    nz_val= list(zip(nz_val[0], nz_val[1]))
    for val in nz_val:
        matrix[val[0], val[1]]-=means_matrix[val[0]]
    return matrix

    

def predict_baseline(matrix, dist_matrix, testing, user_map, movie_map, n,mode,t2,usr_mean,movie_mean):
    pred=[]
    print("Entered Prediction Function")
    overall_mean_movie_rating = matrix.sum()/matrix.count_nonzero()
    print("Overall Mean Movie Rating ",overall_mean_movie_rating)
    no_of_ratings = 0
    no_of_zero = 0
    
   
    testing = testing.head(10000)
    if mode=='user':
        for idx,row in testing.iterrows():
            dist=np.reshape(dist_matrix[:, user_map[row['userId']]], [len(dist_matrix),1])
            
            usr_ratings=t2[:, movie_map[row['movieId']]].todense()
            
            t_rating_dist=list(zip(dist.tolist(), usr_ratings.tolist()))
            t_rating_dist.sort(reverse=True)
            t_rating_dist=t_rating_dist[1:]
            rating = usr_mean[user_map[row['userId']]] + movie_mean[movie_map[row['movieId']]] - overall_mean_movie_rating
            similar_rating = 0
            c = 1
            den = 0
            for i  in range(len(t_rating_dist)):
                if c>=n:
                    break
                elif t_rating_dist[i][1][0]!=0:
                    similar_rating+=(t_rating_dist[i][1][0]+overall_mean_movie_rating)*t_rating_dist[i][0][0]
                    den+=t_rating_dist[i][0][0]
                c+=1
                
            if den==0:
                den=1
            rating+=similar_rating/den
            if rating>5:
                rating=5
            if rating<0:
                rating = usr_mean[user_map[row['userId']]] + movie_mean[movie_map[row['movieId']]] - overall_mean_movie_rating
            pred.append(rating)
    else:
        print(t2.shape)
        for idx,row in testing.iterrows(): 
            dist=np.reshape(dist_matrix[:, movie_map[row['movieId']]], [len(dist_matrix),1])
            movie_ratings=t2[:, user_map[row['userId']]].todense()
            t_rating_dist=list(zip(dist.tolist(), movie_ratings.tolist()))
            t_rating_dist.sort(reverse=True)
            t_rating_dist=t_rating_dist[1:]
            no_of_ratings+=1
            rating=0
            c=1
            den=0
            for i in range(len(t_rating_dist)):
                if c>=n:
                    break
                elif t_rating_dist[i][1][0]!=0:
                    rating+=t_rating_dist[i][1][0]*t_rating_dist[i][0][0]
                    den+=t_rating_dist[i][0][0]
                c+=1
            if den==0:
                den=1
            rating=rating/den
            
            if rating<=0:
                rating = usr_mean[user_map[row['userId']]] + movie_mean[movie_map[row['movieId']]] - overall_mean_movie_rating
                no_of_zero+=1
            pred.append(rating)
    return np.array(pred)
                


# User-user collaborative filtering
print ('collaborative filtering User-User....')
print(type(training))

c=Counter(training.nonzero()[0])

count_movie =  Counter(training.nonzero()[1])

means_matrix=np.squeeze(np.sum(np.array(training.todense()), axis=1))
movie_matrix=np.squeeze(np.sum(np.array(training.todense()), axis=0))

for i in range(means_matrix.shape[0]):
        if i in list(c.keys()):
            means_matrix[i]=means_matrix[i]/c[i]
        else:
            means_matrix[i]=0
for i in range(movie_matrix.shape[0]):
        if i in list(count_movie.keys()):
            movie_matrix[i]=movie_matrix[i]/count_movie[i]
        else:
            movie_matrix[i]=0



t=dc(training)
t2=dc(training)

mask= t!=0
nz_val=np.array(np.nonzero(t))
nz_val= list(zip(nz_val[0], nz_val[1]))

t_start_time=time()
print(len(nz_val))
for val in nz_val:
    t2[val[0],val[1]] = t2[val[0],val[1]] - means_matrix[val[0]] - movie_matrix[val[1]]

means_matrix=np.squeeze(means_matrix)
movie_matrix=np.squeeze(movie_matrix)

user_dist = 1-pairwise_distances(sub_mean(t), metric='cosine')
start_time_item = time()
predictions_usr=predict_baseline(training, user_dist, testing, user_map, movie_map, 10,'user',t2,means_matrix,movie_matrix)
predictions_usr=np.squeeze(predictions_usr)
print('Total time for User-User:', time()- start_time_item)
print('RMSE:', evaluation.RMSE(predictions_usr, truth[0:10000]))
print('spearman_rank_correlation:', evaluation.spearman_rank_correlation(predictions_usr, truth[0:10000]))
print('Precision on top K:' , evaluation.top_k_precision(predictions_usr, testing.head(10000), means_matrix, user_map))


print('collaborative filtering for....')
start_time_item=time()
item_dist=1-pairwise_distances(sub_mean(training.T), metric='cosine')
print('Time taken to calculate distances:', time()-start_time_item)
t2 = t2.T
predictions_mov=predict_baseline(training.T, item_dist, testing, user_map, movie_map, 10,'item',t2,means_matrix,movie_matrix)
predictions=np.squeeze(predictions_mov)
print('Total time for Item-item:', time()- start_time_item)
print('RMSE:', evaluation.RMSE(predictions, truth[0:10000]))
print('spearman_rank_correlation:', evaluation.spearman_rank_correlation(predictions, truth[0:10000]))
print('Precision on top K:' , evaluation.top_k_precision(predictions, testing.head(10000), movie_matrix, movie_map, 5, False))


