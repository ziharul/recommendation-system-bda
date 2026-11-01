"""Pemeriksaan offline dataset dan model dasar tanpa menjalankan server Streamlit."""
from pathlib import Path
from io import BytesIO
import ast
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

root=Path(__file__).resolve().parent
assert (root/'app.py').exists()
ast.parse((root/'app.py').read_text(encoding='utf-8'))
rows=pd.read_csv(root/'data/groceries.csv',header=None,dtype=str)
assert len(rows)==9835, len(rows)
assert rows.shape[1]>=30
movies=pd.read_csv(root/'data/movie.csv')
ratings=pd.read_csv(root/'data/rating.csv')
assert {'title','genres','movieId'}.issubset(movies)
assert {'userId','movieId','rating'}.issubset(ratings)
text=movies.loc[movies.genres!='(no genres listed)','genres'].str.replace('|',' ',regex=False)
X=TfidfVectorizer(stop_words='english').fit_transform(text)
assert len(linear_kernel(X[0],X).ravel())==len(text)

books=pd.read_csv(root/'data/book_knn_interactions.csv.gz')
assert {'userID','bookTitle','bookRating'}.issubset(books)
assert len(books)==54731
pivot=books.pivot_table(index='bookTitle',columns='userID',values='bookRating',aggfunc='mean',fill_value=0)
assert len(pivot)==746, len(pivot)
assert len(pivot.columns)==734, len(pivot.columns)
sparse=csr_matrix(pivot.to_numpy(dtype=np.float32))
model=NearestNeighbors(metric='cosine',algorithm='brute').fit(sparse)
dists, indices=model.kneighbors(sparse[25],n_neighbors=6)
assert len(indices[0])==6

content=['Action sci-fi hacker virtual reality computer illusion','Action sci-fi dream mind control heist illusion','Sci-fi space travel black hole time relativity','Romance love story drama beautiful couple memory','Romance drama ship tragedy beautiful love ocean']
C=TfidfVectorizer(stop_words='english').fit_transform(content)
sim=cosine_similarity(C)
assert sim.shape==(5,5)

sample=[['Roti', 'Susu', 'Mentega'],['Roti', 'Mentega'],['Roti', 'Susu', 'Selai'],['Apel', 'Pisang'],['Roti', 'Susu', 'Mentega', 'Selai']]
from collections import Counter
from itertools import combinations
freq=Counter()
for trans in sample:
 for r in range(1,len(trans)+1):
  for s in combinations(sorted(trans),r):freq[s]+=1
frequent={k:v for k,v in freq.items() if v/len(sample)>=0.4}
assert len(frequent)==11,len(frequent)
print('PASS — sintaks aplikasi valid')
print('PASS — groceries',len(rows),'transaksi')
print('PASS — MovieLens',len(movies),'film /',len(ratings),'rating')
print('PASS — Book-Crossing',len(books),'interaksi /',len(pivot),'buku /',len(pivot.columns),'pengguna')
print('PASS — TF-IDF, cosine similarity, dan kNN berhasil dihitung')
print('PASS — Apriori lima transaksi mengandung 11 frequent itemsets (verifikasi manual)')
