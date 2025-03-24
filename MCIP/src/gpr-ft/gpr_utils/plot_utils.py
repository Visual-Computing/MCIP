import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms as pth_transforms
from PIL import Image
from sklearn.manifold import TSNE
import numpy as np
import torch
from evaluation.utils import get_sorted_distances
import matplotlib.patches as patches

def plot_single(ax, img, cat, t, border=None):

    if type(img) == np.str_:
        img = Image.open(img)
        img = img.convert('RGB')

    ax.imshow(t(img))
    ax.axis('off')
    ax.set_title(str(cat)[:8])

    if border is not None:
        color = "b" if border else "r"
        rect = patches.Rectangle((0,0),223, 223, linewidth=8, edgecolor=color,facecolor='none')
        ax.add_patch(rect)
            
      


def plot_nn(q_image, db_images, q_cat, db_cats, save_path=None, q_predictions=None, title="", scale=1):
    
    t = pth_transforms.Compose([
             pth_transforms.Resize((256, 256), interpolation=3),
             pth_transforms.CenterCrop(224),
        ])
    
    cols = len(db_images[0])+1
    rows = len(q_image)

    if q_predictions is None:
        q_title_fn = lambda i: q_cat[i]
    else:
        q_title_fn = lambda i: f"{q_cat[i]}({q_predictions[i][0]})"

    fig = plt.figure(figsize=(int(scale*cols*2), 5+int(rows*2.5)))
    fig.suptitle(title, fontsize=50)
    for i in range(0, len(q_image), 1):
        ax = plt.subplot(rows, cols, i*cols + 1) # y*width+0 + 1, since ax starts counting at 1
        plot_single(ax, q_image[i], q_title_fn(i), t)
        for j in range(0, len(db_images[0]), 1):
            ax = plt.subplot(rows, cols, cols*i+(j+1) + 1)
            image_cat = db_cats[i, j]
            border = None if image_cat == "" else (image_cat == q_cat[i])
            plot_single(ax, db_images[i, j], image_cat, t, border=border)

    plt.subplots_adjust(left=0.0,
                    bottom=0.0, 
                    right=1, 
                    top=1, 
                    wspace=0.1, 
                    hspace=0.1)
                    
    if save_path is not None:
        fig.savefig(save_path)
        plt.close()
    else: 
        return fig

def find_and_plot_nn(image_paths, embeddings, cats, save_path, k=20, n_querries=30, skip_self=1, title="", device="cpu"):

    skip = int(len(cats) / n_querries)
    embeddings = embeddings.astype(np.float32)
    q_embds = embeddings[::skip]
    _, f_indices = get_sorted_distances(embeddings, q_embds, k=k+skip_self, device=device)

    q_images = image_paths[::skip]
    q_cats = cats[::skip]

    db_images = []
    db_cats = []
    for nn_index_row in f_indices:
        db_images.append(image_paths[nn_index_row[skip_self:]])
        db_cats.append(cats[nn_index_row[skip_self:]])

    db_images = np.array(db_images)
    db_cats = np.array(db_cats)
    plot_nn(q_images, db_images, q_cats, db_cats, save_path, title=title)



def plot_anchor_tsne(anchors, vectors, skip=8, title=""):
    
    landmarks_i = 81313
    ali_i = landmarks_i+50019
    iNat = ali_i + 10000
    im_i = iNat+100480 - 81313
    faces_i = im_i+8631
    
    colors = np.zeros((faces_i,))
    
    colors[landmarks_i:ali_i] = 1 # ali
    colors[ali_i:iNat] = 4 # iNat
    colors[iNat:im_i] = 2 # imageNet
    colors[im_i:] = 3 # faces
    
    anchors = anchors[::skip]
    colors = colors[::skip]
    
     # dists
    sims = torch.matmul(torch.tensor(anchors), torch.tensor(vectors.T))
    nn_indices = torch.argsort(sims, dim=1, descending=True)[:,0].numpy()
    #print(sims[nn_indices].shape, sims[nn_indices][:,0])
    
    embeddings = TSNE(init="pca", random_state=777, verbose=0).fit_transform(anchors)
    
    fig = plt.figure(figsize=(8*3,6))
    #cmap = plt.cm.get_cmap(lut=4)
    ax1 = plt.subplot(1, 3, 1)
    cmap = LinearSegmentedColormap.from_list("", ["red","blue","yellow", "black", "green"],N=5)
    im = ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap, s=20)
    cbar = plt.colorbar(im, ax=ax1, cmap=cmap, ticks=[0.4, 0.4+0.8, 0.4+1.6, 0.4 + 2.4, 0.4+3.2])
    cbar.ax.set_yticklabels(['Landmarks', "AliP", 'ImageNet','Faces', "iNat"])
    
    
    
    ax3 = plt.subplot(1, 3, 2)
    colors = (np.arange(len(vectors)) / 2000).astype(int)[nn_indices]
    
    cmap = plt.cm.get_cmap(lut=6)
    im = ax3.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap)
    cbar = plt.colorbar( im, ax=ax3, cmap=cmap)
    cbar.ax.set_yticklabels(['LM', 'iNat', 'Sketch', "INSTRE", "SOP", "Faces"]);
    
    
    embeddings = TSNE(init="pca", random_state=777, verbose=0).fit_transform(vectors)
    colors = (np.arange(len(vectors)) / 2000).astype(int)
    
    ax2 = plt.subplot(1, 3, 3)
    
    cmap = plt.cm.get_cmap(lut=6)
    im = ax2.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap)
    cbar = plt.colorbar( im, ax=ax2, cmap=cmap)
    cbar.ax.set_yticklabels(['LM', 'iNat', 'Sketch', "INSTRE", "SOP", "Faces"]);
    

    plt.title(title)
    
    return fig