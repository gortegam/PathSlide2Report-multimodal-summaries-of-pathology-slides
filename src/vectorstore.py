"""
vectorstore.py
Simple wrapper around FAISS for storing and retrieving embeddings + metadata.
"""

import faiss
import numpy as np
import pickle

class FaissStore:
    def __init__(self, dim, index_path=None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []
        self.index_path = index_path

    def add(self, embeddings, meta_list):
        emb = np.vstack(embeddings).astype("float32")
        self.index.add(emb)
        self.metadata.extend(meta_list)

    def search(self, query_emb, k=5):
        D, I = self.index.search(query_emb.astype("float32"), k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

    def save(self):
        if self.index_path:
            faiss.write_index(self.index, self.index_path + ".index")
            with open(self.index_path + ".meta", "wb") as f:
                pickle.dump(self.metadata, f)

    def load(self):
        if self.index_path:
            self.index = faiss.read_index(self.index_path + ".index")
            with open(self.index_path + ".meta", "rb") as f:
                self.metadata = pickle.load(f)
