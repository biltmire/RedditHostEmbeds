import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm

def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_word_vector(sent, word, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    idx = get_word_idx(sent,word)

    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, token_ids_word, model, layers)

def get_sentence_vector(sent, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    return output.mean(dim=0)

def cos(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def main(layers=None):

    INPUT_FILE = '../data/csv/desc_list_0_10000.csv'
    OUTPUT_FILE = '../embeddings/desc_vecs_0_10000.csv'
    META_FILE = '../embeddings/desc_meta_0_10000.csv'

    # Use last four layers by default
    layers = [-4, -3, -2, -1] if layers is None else layers
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    descs = pd.read_csv(INPUT_FILE)

    descs = descs[~descs.desc.isna()]

    vector_list = []

    for desc in tqdm(descs['desc'].values):
        desc_vec = get_sentence_vector(desc,tokenizer,model,layers).cpu().detach().numpy()
        vector_list.append(desc_vec)

    vector_df = pd.DataFrame(vector_list)

    vector_df.to_csv(OUTPUT_FILE,index=False,header=False,sep='\t')
    #Normalize vectors
    vector_df = vector_df.divide(np.linalg.norm(vector_df, axis=1), axis=0)
    descs.to_csv(META_FILE,index=False,header=True,sep='\t')

if __name__ == '__main__':
    main()
