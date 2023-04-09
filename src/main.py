#
#
#  
#
#

import os
import settings

import pandas as pd
import numpy as np
import openai

from openai.embeddings_utils import get_embedding

openai.api_key = settings.OPENAI_API_KEY

def get_data():
    cur_dir = os.getcwd()
    parent_dir = os.path.dirname(cur_dir)
    
    return pd.read_csv(os.path.join(parent_dir, 'data', 'actions_data.csv'))

def embed_data(df, embedding_model = 'text-embedding-ada-002'):
    
    df['embeddings'] = df.action.apply(lambda x: get_embedding(x, engine=embedding_model))
    df.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'actions_data_embedded.csv'))
    
def read_embeddings():
    
    df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'actions_data_embedded.csv'))
    df['embeddings'] = df.embeddings.apply(eval).apply(np.array)
    df = df.loc[:, ['action', 'embeddings']]
    
    return df

def main():
    
    print(read_embeddings())
    

    
    
if __name__ == '__main__':
    main()