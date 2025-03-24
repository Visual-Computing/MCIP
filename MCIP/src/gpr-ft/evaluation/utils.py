import numpy as np
import torch
import torch.nn.functional as F
#from evaluation.faiss_utils import get_sorted_distances

from tqdm import tqdm

def get_sorted_distances(features_DB, features_Q=None, k=None, device="cuda"):

    """
    Computes the cosine similarity of embeddings and returns
    similarities and distances of k nearest neighbours
    
    Parameters
    ----------
    features_DB : array-like, shape = [n_samples, dimensionality]
        Database feature vectors
    features_Q : array-like, shape = [n_samples, dimensionality]
        Query feature vectors. If this parameter is not given, the database fv´s are used as querries
    k : int
        k nearest neighbours
    """

    if type(features_Q) == type(None):
        features_Q = features_DB

    if k is None or k is np.inf:
        k = len(features_DB)

    num_test_images = features_Q.shape[0]
    num_chunks = 1000 if num_test_images > 100000 else 5
    imgs_per_chunk = min(1, num_test_images // num_chunks)

    db_features = torch.from_numpy(features_DB).float().to(device).t()
    
    all_indices, all_sims = [], []
    with torch.no_grad():
       
        for idx in tqdm(range(0, num_test_images, imgs_per_chunk), desc="Computing Nearest Neighbours"):
            # get the features for test images and normalize the features if needed
            batch_features = features_Q[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]
            batch_features = torch.from_numpy(batch_features).float().to(device)

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(batch_features, db_features)
            distances, indices = similarity.topk(
                k, largest=True, sorted=True
            )
            all_indices += list(indices.cpu().numpy())
            all_sims += list(distances.cpu().numpy())

    return np.array(all_sims), np.array(all_indices)

def get_average_precision_score(y_true, k=None):
    """
    Average precision at rank k
    Modified to only work with sorted ground truth labels
    From: https://gist.github.com/mblondel/7337391
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Binary ground truth (True if relevant, False if irrelevant), sorted by the distances. 
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    if k is None:
        k = np.inf
    
    n_positive = np.sum(y_true.astype(np.int32) == 1)
    
    if n_positive == 0:
        # early return in cases where no positives are among the ranks
        return 0
    
    y_true = y_true[:min(y_true.shape[0], k)].astype(np.int32)
    
    score = 0
    n_positive_seen = 0
    pos_indices = np.where(y_true == 1)[0]
    
    for i in pos_indices:
        n_positive_seen += 1
        score += n_positive_seen / (i + 1.0)
    
    return score / n_positive


def compute_mean_average_precision(categories_DB, 
                                    features_DB=None, 
                                    features_Q=None, 
                                    categories_Q=None, 
                                    indices=None, 
                                    k=np.inf):
    """
    Performs a search for k neirest neighboors with the specified indexing method and computes the mean average precision@k 
    
    Parameters
    ----------
    features_DB : array-like, shape = [n_samples, dimensionality]
        Database feature vectors
    features_Q : array-like, shape = [n_samples, dimensionality]
        Query feature vectors. If this parameter is not given, the database fv´s are used as querries
    categories_DB : array-like, shape = [n_samples_DB]
        Database categories
    categories_Q : array-like, shape = [n_samples_Q]
        Query categories. If this parameter is not given, the database categories are used
    indices: array-lile, shape = [n_samples_Q, n_samples_DB]
        Nearest neighbours indices 
    k : int
        Mean average precision at @k value. If np.inf, this function computes the mean average precision score
    Returns
    -------
    Mean average precision @k : float
    """

    if (indices is None) & (features_DB is None):
        raise ValueError("Either indices or features_DB has to be provided ")
    
    if features_Q is None: features_Q = features_DB
    if categories_Q is None: categories_Q = categories_DB
    
    if (indices is None):
        _, indices = get_sorted_distances(features_DB, features_Q, k=k)
    
    aps = []
    for i in range(0, len(indices)):
        aps.append(get_average_precision_score((categories_DB[indices[i]] == categories_Q[i]), k))
    
    return aps

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

def compute_recalls(X, T, recalls=[1,2,4,8]):

    # get predictions by assigning nearest 8 neighbors with cosine
    X = torch.tensor(X)
    T = torch.tensor(T)
    
    K = np.max(recalls)
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in recalls:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        #print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return np.array(recall)


def find_kNN(embs_q, embs_db, k, val_list=None, skip_self=0, device="cpu"):

    #skip_self has to be 1 if query can be in db

    _, indices = get_sorted_distances(embs_db, embs_q, k=k+skip_self, device=device)
    
    if val_list is None:
        return indices
    else:
        r_t = []

        for v in val_list: # get NNs for each list in val_list
            v_r = [] #this will be returned
            for nn_index_row in indices:
                v_r.append(v[nn_index_row[skip_self:]])
        
            r_t.append(np.array(v_r))
        
        return r_t

def compute_accuracy(labels, preds, n_float=4):
    return np.mean((labels == preds).astype(np.float)).round(n_float)

def reduce_kNNs(list_vals, indices, n_rows=30, n_cols=20):

    r_t = []
    skip = int(len(indices) / n_rows)
    r_indices = indices[::skip, :n_cols]
    for val in list_vals:
        v_r = [] #this will be returned
        for nn_index_row in r_indices:
            v_r.append(val[nn_index_row])

        r_t.append(np.array(v_r))

    return skip, r_t

def nearest_neighbor_test(train_features, test_features, train_labels, test_labels, device, temperature=0.1, k=21):
    #https://github.com/mgwillia/unsupervised-analysis/blob/main/experiments/knn_classifier.py
    ############################################################################
    # Step 1: get train and test features

    num_classes = len(np.unique(train_labels))
    train_features = torch.from_numpy(train_features).float().to(device).t()
    train_labels = torch.LongTensor(train_labels).to(device)
    ###########################################################################
    # Step 2: calculate the nearest neighbor and the metrics
    top1, top5, total = 0.0, 0.0, 0
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks

    all_indices = []
    all_preds = []
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(k, num_classes).to(device)
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images and normalize the features if needed
            features = test_features[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]#, :]
            batch_size = targets.shape[0]
            features = torch.from_numpy(features).float().to(device)
            targets = torch.LongTensor(targets).to(device)

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            distances, indices = similarity.topk(
                k, largest=True, sorted=True
            )
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(temperature).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)

            all_indices += list(indices.cpu().numpy())
            all_preds += list(predictions.narrow(1, 0, 1).cpu().numpy())

    top1 = np.round(top1  / total, 4)
    top5 = np.round(top5 / total, 4)

    return np.array(all_indices), np.array(all_preds), top1, top5


def compute_mean_rank_and_topK(query_features, db_features, query_targets, db_vid_numbers, topK=100):
    
    _, indices = get_sorted_distances(db_features, query_features)
    
    ranks = []
    is_in_top100 = []
    for i in range(len(indices)):
        target = query_targets[i]
        
        first_index_of_target = np.where(db_vid_numbers[indices[i]] == target)[0][0]
        
        #print(query_targets[i],db_vid_numbers[indices[i]][first_index_of_target], first_index_of_target, db_vid_numbers[indices[i]][:10])
        ranks.append(first_index_of_target)
        is_in_top100.append(first_index_of_target < topK)
    
    return np.mean(ranks),  np.mean(is_in_top100)