import torch
import numpy as np
import torch.nn.functional as F

def l2_norm(x):
    x = x / torch.norm(x, dim=-1, keepdim=True)
    return x

def binarize(T, nb_classes, device):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(device)
    return T

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, device, mrg = 0.1, alpha = 32, eps=0.0):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).to(device))
        torch.nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.eps = eps
        self.device = device
        
    def forward(self, X, T):
        P = self.proxies
       
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes, device=self.device)
        P_one_hot = P_one_hot.mul_(1 - self.eps).add_(self.eps / self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))*P_one_hot
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))*N_one_hot
        
        greater_then_eps = 1 - self.eps
        
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) >= greater_then_eps).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
       
        
        P_sim_sum = torch.where(P_one_hot >= greater_then_eps, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot <= greater_then_eps, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term 
            
        
        return loss


class SoftmaxCrossEntropy(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, device):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).to(device))
        torch.nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.sce = torch.nn.CrossEntropyLoss()
        self.device = device
        
    def forward(self, X, T):
        P = self.proxies
        cos = F.linear(X, P)  # Calcluate cosine similarity
        target = binarize(T = T, nb_classes = self.nb_classes, device=self.device)
        
        return self.sce(cos, target)


import math

#https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
class ArcMarginProduct(torch.nn.Module):
    """Implementation of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, out_features, in_features, device, s=30.0, m=0.1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.s = s
        #self.m = m
        self.proxies = torch.nn.Parameter(torch.randn(out_features, in_features).to(device))
        torch.nn.init.xavier_uniform_(self.proxies)

        self.easy_margin = easy_margin
        
        self.register_buffer("cos_m", torch.tensor(math.cos(m)))#1
        self.register_buffer("sin_m", torch.tensor(math.sin(m)))#0
        self.register_buffer("th", torch.tensor(math.cos(math.pi - m)))#-1
        self.register_buffer("mm", torch.tensor(math.sin(math.pi - m) * m))#0
        
        self.register_buffer("s", torch.tensor(s))
        self.register_buffer("m", torch.tensor(m))
        
        self.sce = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, input, label):

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.proxies)).float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
       
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
       
        return self.sce(output, one_hot)
    
import math 

class SoftTargetCrossEntropy(torch.nn.Module):
    def forward(self, x, target):
        loss = torch.mean(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelArcMarginProduct(torch.nn.Module):
    # Implementation of the Multi-Label ArcMargin Loss 
    
    def __init__(self, out_features, in_features, device, s=30.0, m=0.1, easy_margin=False, parts=4):
        super(MultiLabelArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.proxies = torch.nn.Parameter(torch.randn(out_features, in_features).to(device))
        torch.nn.init.xavier_uniform_(self.proxies)

        self.easy_margin = easy_margin
        
        self.register_buffer("cos_m", torch.tensor(math.cos(m)))#1
        self.register_buffer("sin_m", torch.tensor(math.sin(m)))#0
        self.register_buffer("th", torch.tensor(math.cos(math.pi - m)))#-1
        self.register_buffer("mm", torch.tensor(math.sin(math.pi - m) * m))#0
        
        self.register_buffer("s", torch.tensor(s))
        self.register_buffer("m", torch.tensor(m))
        
        self.sce = SoftTargetCrossEntropy()
        self.device = device
        self.parts = parts

    def transform_class_list(self, labels):

        reshaped_labels = labels.reshape((-1, self.parts))

        one_hot = torch.zeros((reshaped_labels.shape[0], self.out_features))

        for i in range(len(reshaped_labels)):
            coh = torch.zeros((self.out_features))
            coh[reshaped_labels[i]] = 1
            one_hot[i] = coh
        
        return one_hot.to(self.device)

    def forward(self, input, labels):

        if input.size()[0] > 128:
            cosine = F.linear(F.normalize(input), F.normalize(self.proxies)).float()
            one_hot = torch.zeros(cosine.size()).to(self.device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        else:
            #input.size()[0]
            one_hot = self.transform_class_list(labels)
            # --------------------------- cos(theta) & phi(theta) ---------------------------
            cosine = F.linear(F.normalize(input), F.normalize(self.proxies)).float()

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
       
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

      
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        loss_per_row = torch.sum(-one_hot * F.log_softmax(output, dim=-1), dim=-1)
    
        num_classes_per_row = one_hot.sum(dim = 1)
        loss = loss_per_row / num_classes_per_row
        return loss.mean()

class TextSoftTargetCrossEntropy(torch.nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1) + torch.max(-(1-target) * F.log_softmax(x, dim=-1), dim=-1)[0]*0.1
        return loss.mean()

class MultiLabelArcMarginProductForText(torch.nn.Module):
    
    # Implementation of the Multi-Caption ArcMargin Loss for the MCIP approach
    def __init__(self, in_features, device, s=30, m=0.1, easy_margin=False):
        super(MultiLabelArcMarginProductForText, self).__init__()
        self.in_features = in_features

        self.easy_margin = easy_margin
        
        self.register_buffer("cos_m", torch.tensor(math.cos(m)))#1
        self.register_buffer("sin_m", torch.tensor(math.sin(m)))#0
        self.register_buffer("th", torch.tensor(math.cos(math.pi - m)))#-1
        self.register_buffer("mm", torch.tensor(math.sin(math.pi - m) * m))#0
        
        self.register_buffer("s", torch.tensor(s))
        self.register_buffer("m", torch.tensor(m))
        
        self.sce = TextSoftTargetCrossEntropy() #
        self.device = device

    def transform_class_list(self, labels):

        amount_of_texts_in_batch = torch.cat(labels).shape[0]

        one_hot = torch.zeros((len(labels), amount_of_texts_in_batch))
        
        #for each image in batch
        c = 0
        for i in range(len(labels)):
            coh = torch.zeros((amount_of_texts_in_batch))
            
            len_current = labels[i].shape[0]
            coh[c:c+len_current] = 1 # if first image has 3 texts [1,1,1,0,0,0,....,0]
            one_hot[i] = coh
            c += len_current
        
        return one_hot.to(self.device)

    def forward(self, input_features, padded_text_features, model_fn):
        # i need to include alist of keep indices for each list of text
       
        #convert padded text_features to list
        text_features = []
        for ki, padded_tf in enumerate(padded_text_features):
           
            sum = padded_tf.sum(dim=1)
            
            #is_empty = padded_tf == ""
            keep = (sum != 64) # 64 is the sum of an tokenized empty string
            text_features.append(padded_tf[keep])



        # text_featutes has to be a list of (n, embedding_dim) tuples since we can have n texts for each image
        one_hot = self.transform_class_list(text_features)
        text_features = torch.vstack(text_features)

        text_features = model_fn(text_features)


        cosine = F.linear(F.normalize(input_features), F.normalize(text_features)).float()
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
       
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        one_hot = one_hot / one_hot.sum(dim=1, keepdim=True) # this leads to nan since devision by 0
        one_hot = torch.nan_to_num(one_hot)
        return self.sce(output, one_hot)

    

def get_loss(exp_params, n_classes, emb_size, device, loss_key=None):

    loss_key = exp_params.loss if loss_key is None else loss_key
    loss_module = None
    if loss_key == "PA":
        loss_module = Proxy_Anchor(n_classes, emb_size, device=device, alpha=np.sqrt(emb_size), mrg=0.1).to(device)
    
    elif loss_key == "SCE":
        loss_module = SoftmaxCrossEntropy(n_classes, emb_size, device=device).to(device)

    elif loss_key == "ArcFace":
        loss_module = ArcMarginProduct(n_classes, emb_size, s=exp_params.arcface_scale, m=exp_params.arcface_margin, device=device).to(device)

    elif loss_key == "MultiLabelArcFace":
        loss_module = MultiLabelArcMarginProduct(n_classes, emb_size, s=exp_params.arcface_scale, m=exp_params.arcface_margin, device=device).to(device)
    
    elif loss_key == "TextArcFace":
        loss_module = MultiLabelArcMarginProductForText(emb_size, s=exp_params.arcface_scale, m=exp_params.arcface_margin, device=device).to(device)
    
    return loss_module
