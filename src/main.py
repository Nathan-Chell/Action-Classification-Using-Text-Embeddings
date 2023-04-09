#
#
#  
#
#

import os
import settings

import pandas as pd
from openai.embeddings_utils import get_embedding

os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

def get_data():
    cur_dir = os.getcwd()
    parent_dir = os.path.dirname(cur_dir)
    
    return pd.read_csv(os.path.join(parent_dir, 'data', 'actions_data.csv'))

def embed_data(df, embedding_model):
    
    df['embeddings'] = df.action.apply(lambda x: get_embedding(x, engine=embedding_model))
    df.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'actions_data_embedded.csv'))
    

def main():
    
    embedding_model = 'text-embedding-ada-002'
    embedding_encoding = 'cl100k_base'
    max_tokens = 8000
    
    data = get_data()
    embed_data(data, embedding_model)
    
    
if __name__ == '__main__':
    main()