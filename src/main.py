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
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

openai.api_key = settings.OPENAI_API_KEY

def get_data():
    cur_dir = os.getcwd()
    parent_dir = os.path.dirname(cur_dir)
    
    return pd.read_csv(os.path.join(parent_dir, 'data', 'actions_data.csv'))

def embed_data(df, embedding_model = 'text-embedding-ada-002'):
    
    df['embeddings'] = df.action.apply(lambda x: get_embedding(x, engine=embedding_model))
    df.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'actions_data_embedded.csv'))
    
    return df

def read_embeddings():
    
    df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'actions_data_embedded.csv'))
    df['embeddings'] = df.embeddings.apply(eval).apply(np.array)
    df = df.loc[:, ['action', 'embeddings']]
    
    return df

def Plot_clusters(matrix, df):
    
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    for category, color in enumerate(["purple", "green", "red"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Clusters identified visualized in language 2d using t-SNE")
    plt.show()
    
def Cluster(matrix, df):
    
    n_clusters = 3
    
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df["Cluster"] = labels

    
    Plot_clusters(matrix, df)
    
    print(df.loc[:, ['action', 'Cluster']])
    
def main():
    
    #df = read_embeddings()
    data = get_data()
    df = embed_data(data)
    matrix = np.vstack(df.embeddings.values)
    Cluster(matrix, df)

    
    
if __name__ == '__main__':
    main()