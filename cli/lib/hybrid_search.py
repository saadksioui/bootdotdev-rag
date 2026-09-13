import os
from collections import defaultdict
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_search = self._bm25_search(query, limit)
        semantic_search = self.semantic_search.search_chunks(query, limit)
        document_scores = defaultdict(dict)
        for item in bm25_search:
            idx = item[0]
            score = item[1]
            document_scores[idx]['bm25_score'] = score
        for item in semantic_search:
            idx = item['id']
            score = item['score'].item()
            document_scores[idx]['semantic_score'] = score
        for idx, item in document_scores.items():
            bm25_score = item.get('bm25_score', 0.0)
            semantic_score = item.get('semantic_score', 0.0)
            hybrid_score = alpha * bm25_score + (1 - alpha) * semantic_score
            document_scores[idx]['hybrid_score'] = hybrid_score
        return sorted(document_scores.items(), key=lambda item:item[1]['hybrid_score'], reverse=True)

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_search = self._bm25_search(query, limit)
        semantic_search = self.semantic_search.search_chunks(query, limit)
        document_ranks = defaultdict(dict)
        for rank, item in enumerate(bm25_search, start=1):
            idx = item[0]
            document_ranks[idx]['bm25_rank'] = rank
            document_ranks[idx]['rrf_score'] = document_ranks[idx].get('rrf_score', 0.0) + rrf_score(rank)
        for rank, item in enumerate(semantic_search, start=1):
            idx = item['id']
            document_ranks[idx]['semantic_rank'] = rank
            document_ranks[idx]['rrf_score'] = document_ranks[idx].get('rrf_score', 0.0) + rrf_score(rank)
        return sorted(document_ranks.items(), key=lambda item:item[1]['rrf_score'], reverse=True)


def normilaze(nums):
    if not nums:
        return []
    min_nums = min(nums)
    max_nums = max(nums)
    if min_nums == max_nums:
        return [1.0 for _ in range(len(nums))]
    res = []
    for score in nums:
        norm = (score - min_nums) / (max_nums - min_nums)
        res.append(norm)
    return res


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)