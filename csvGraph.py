import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch_geometric.data import download_url, extract_zip

# url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# extract_zip(download_url(url, '.'), '.') # 下载并解压

movie_path = './ml-latest-small/movies.csv'
rating_path = './ml-latest-small/ratings.csv'

import pandas as pd

# print(pd.read_csv(movie_path).head())
# print(pd.read_csv(rating_path).head())

import torch

def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    """
    df=pd.read_csv(rating_path).head(5)
	df.index # RangeIndex(start=0, stop=5, step=1)
	df.index.unique() # Int64Index([0, 1, 2, 3, 4], dtype='int64')
    """
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        # encoders.itmes() 每次迭代返回元组对象(key,element)
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

from sentence_transformers import SentenceTransformer

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device="cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad() # 禁用梯度, 前向计算节省显存
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

class GenresEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping)) # [num_movies, num_genres]
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


movie_x, movie_mapping = load_node_csv(
    movie_path, index_col='movieId', encoders={
        'title': SequenceEncoder(),
        'genres': GenresEncoder()
    })

# print(movie_x, movie_mapping)
# exit()
_, user_mapping = load_node_csv(rating_path, index_col='userId')

from torch_geometric.data import HeteroData

data = HeteroData()

data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
data['movie'].x = movie_x

print(data)
"""
HeteroData(
  user={ num_nodes=610 },
  movie={ x[9742, 404] }
)
"""
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


edge_index, edge_label = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

data['user', 'rates', 'movie'].edge_index = edge_index
data['user', 'rates', 'movie'].edge_label = edge_label

print(data)
"""
HeteroData(
  user={ num_nodes=610 },
  movie={ x=[9742, 404] },
  (user, rates, movie)={
    edge_index=[2, 100836],
    edge_label=[100836, 1]
  }
)
"""
