import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    문서 로드 → 임베딩 → FAISS 인덱싱 → 검색 기능 제공
    """
    def __init__(self,
                 doc_path: str = "data/emergency.jsonl",
                 idx_path: str = "data/faiss_index.idx",
                 meta_path: str = "data/docs_meta.pkl",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.doc_path = doc_path
        self.idx_path = idx_path
        self.meta_path = meta_path
        self.model = SentenceTransformer(model_name)

        if os.path.exists(self.idx_path) and os.path.exists(self.meta_path):
            self._load_index()
        else:
            self._build_index()

    def _load_index(self):
        self.index = faiss.read_index(self.idx_path)
        with open(self.meta_path, 'rb') as f:
            self.documents = pickle.load(f)
        self.vector_dim = self.index.d
        print(f"[Retriever] Loaded FAISS index ({self.index.ntotal} docs, dim={self.vector_dim})")

    def _build_index(self):
        # 문서 로드
        self.documents = []
        texts = []
        with open(self.doc_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.documents.append(data)
                # title + content 합치기
                texts.append(f"{data['title']}\n{data['content']}")

        # 임베딩 생성
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.vector_dim = embeddings.shape[1]

        # FAISS 인덱스 생성 및 데이터 추가
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.index.add(embeddings)

        # 인덱스 및 메타데이터 저장
        os.makedirs(os.path.dirname(self.idx_path), exist_ok=True)
        faiss.write_index(self.index, self.idx_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"[Retriever] Built FAISS index ({self.index.ntotal} docs, dim={self.vector_dim})")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        query를 임베딩하여 가장 유사한 상위 top_k 문서 반환
        :returns: list of dict (원본 JSONL 문서)
        """
        q_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])
        return results
    


def load_disease_titles(file_path="data/emergency.jsonl"):
    disease_titles = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            disease_titles.append(data["title"])
    return disease_titles

# 질환명이 질문에 포함되어 있는지 확인하는 함수
def is_disease_in_query(query: str, disease_titles: list) -> bool:
    # 질환명이 질문에 포함되어 있는지 확인
    return any(disease.lower() in query.lower() for disease in disease_titles)



# 전역 인스턴스 생성
retriever = Retriever()