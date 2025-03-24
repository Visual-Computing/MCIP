import numpy as np
from evaluation.utils import get_sorted_distances
import torch
import os
import concurrent.futures


def find_unique_indices(string_list):
    seen = set()
    unique_indices = [i for i, string in enumerate(string_list) if string not in seen and not seen.add(string)]
    return np.array(unique_indices)



def create_pseudo_captions(exp_params):

    caption_value_files = exp_params.data.caption_value_files.split(";")
    text_values = np.concatenate([np.load(f) for f in caption_value_files])

    text_embedding_files = exp_params.data.caption_embeddings_files.split(";")
    text_embeddings = torch.cat([torch.load(f).reshape((-1, 768)) for f in text_embedding_files], dim=0).half().numpy()

    train_ds_image_embeddings = np.load(exp_params.data.train_image_embeddings_file_path).astype(np.float16)
    unique_text_indxs = find_unique_indices(text_values)
    text_values = text_values[unique_text_indxs]
    text_embeddings = text_embeddings[unique_text_indxs]

    print("Starting to create pseudo captions... computing similarites on", exp_params.gpu_id)
    print("Number of unique texts: ", len(text_values))
    print("Number of unique text embeddings: ", len(text_embeddings))
    print("Number of train image feautres: ", len(train_ds_image_embeddings))
    s_sims, s_indices = get_sorted_distances(text_embeddings, train_ds_image_embeddings, k=20, device=exp_params.gpu_id)
    
    image_file_paths = np.load(exp_params.data.image_paths)
    caption_output_dir = exp_params.data.caption_output_dir
    print("Similarities computed", caption_output_dir, s_sims.shape, s_indices.shape, len(image_file_paths))

    os.makedirs(os.path.dirname(caption_output_dir), exist_ok=True)
    def find_and_save_texts(i, sim_th=0.271):

        sims = s_sims[i]
        image_file_name = image_file_paths[i]
        
            
        include_indxs = np.where(sims > sim_th)[0]
        texts = text_values[s_indices[i][include_indxs]]
        out = texts
        
        save_file_name = os.path.join(caption_output_dir, os.path.basename(image_file_name).split(".")[0] + f"_text_{sim_th:.3f}")

        
        np.savez_compressed(save_file_name, out)

        if i % 20000 == 0: 
            print(i, save_file_name)
            print(out)
            print("---")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Map the function to the range of indices (0 to len(s_sims))
        executor.map(find_and_save_texts, range(len(s_sims)))

