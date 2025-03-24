import faiss 
import numpy as np
from enum import Enum
import warnings

INDEX_TYPES = Enum('Index_Types', 'Dot L2')

def _generate_index_options_message():
    """
    Generate ','-separated string for display of available index types
    
    Returns
    ----------
    message: string
    """
    return ', '.join(map(lambda enum_value: str(enum_value).split('.')[1], INDEX_TYPES))

def _get_faiss_index(gpu, index_type, feature_dim):
    """
    Generate a  Faiss Index object of desired type
    
    Parameters
    ----------
    gpu : boolean
        If True, a Faiss GPU index object is build. Note that this option limits 
    index_type : faiss_utils.INDEX_TYPES Enum
        Sets the distance metric used when performing future searches with the new index
        
    Returns
    ----------
    index: Faiss.Index Object
        
    """
    
    if index_type == INDEX_TYPES.L2:
        index = faiss.IndexFlatL2(feature_dim)
        
    elif index_type == INDEX_TYPES.Dot:
        index = faiss.IndexFlat(feature_dim)
    else:
        raise ValueError('Invalid type of index selected. Valid options are: ' + _generate_index_options_message())

    if gpu:
        gpu_resource = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
    
    return index


def get_average_precision_score(y_true, k=np.inf):
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

def get_sorted_distances(features_DB, features_Q=None, k=np.inf, index_type=INDEX_TYPES.L2, gpu=False):
    """
    Builds a Faiss index of desired type, indexes the given features as database features and performs a search.
    Faiss returns a matrix of sorted distances and the corresponding indices in the input features list. 
    These values are returned by this function
    
    Parameters
    ----------
    features_DB : array-like, shape = [n_samples_DB, dimensionality]
        Database feature vectors
    features_Q : array-like, shape = [n_samples_Q, dimensionality]
        Query feature vectors. If this parameter is not given, the database fv´s are used as querries
    k : int
        Number of neirest neighbors to search for.
    index_type : faiss_utils.INDEX_TYPES Enum
        Sets the distance metric used for this search
    gpu: boolean
        Enables GPU accelerated distance calculations and search.
        Note: If True, this parameter limits k to k=1024 due to Faiss GPU limitations
    Returns
    -------
    sorted_distances : array-like, shape = [n_features_A, k]
        A sorted distance matrix, where each row i represents distances of the k neirest neighbors to entity at features_DB[i]
    indices: array-like, shape = [n_features_A, k]
        Ranking order of indices sorted by the distances
    """
    if not features_DB.flags.contiguous:
        features_DB = np.ascontiguousarray(features_DB)
    
    if k == np.inf: k = features_DB.shape[0]
        
    if features_Q is None: 
        features_Q = features_DB
    else:
        if not features_Q.flags.contiguous:
            features_Q = np.ascontiguousarray(features_Q)
        
    # create desired faiss index
    index_flat = _get_faiss_index(gpu, index_type, features_Q.shape[-1])
    
    # add all feature vectors to as database entities
    
    index_flat.add(features_DB)
    
    # since gpu indices only support a search of up to 1024 neighboors k is limited to 1024
    if gpu & (k > 1024):
        sorted_distances, indices = index_flat.search(features_Q, 1024)
        warnings.warn('This search is limited to 1024 neirest neighbors due to Faiss GPU limitation')
        return sorted_distances, indices
    
    sorted_distances, indices = index_flat.search(features_Q, k)
    return sorted_distances, indices
