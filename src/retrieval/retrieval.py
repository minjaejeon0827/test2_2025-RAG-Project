# 소스코드 분석 (2026.05.27 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 드디어 앞서 retrieval_main.py가 호출하던 retrieve_documents의 실제 구현이네요! 이 파일이 RAG의 R(검색)의 핵심 두뇌예요. 여기엔 검색 분야의 중요한 개념(유사도 검색, 하이브리드 검색, 재정렬)이 다 들어있어서 차근차근 풀어드릴게요.

# 🗺️ 전체 RAG 파이프라인에서의 위치 (이제 거의 완성!)
# 지금까지 분석한 파일들을 모두 연결하면:

# main.py
#    │
#    ├─ loader_main.py → data_loader.py, splitter.py   ✅ (문서 준비)
#    │
#    ├─ embedding_main.py → vector_db.py                (벡터DB 구축)
#    │
#    ├─ retrieval_main.py → retrieval.py                ⬅️ 지금 여기 완성!
#    │     ├─ retrieve_documents (similarity/hybrid)
#    │     └─ rerank_documents (CrossEncoder)
#    │
#    └─ hf_generator.py / openai_generator.py           ✅ (답변 생성)

# 이제 RAG의 흐름이 거의 다 보이시죠? 문서 준비 → 벡터화 → 검색+재정렬 → 답변 생성. 이 파일이 찾아낸 정확한 문서들이 generator의 프롬프트로 들어가 최종 답변이 만들어집니다.
# 궁금한 부분(예: BM25 알고리즘이 점수를 어떻게 계산하는지, EnsembleRetriever가 두 점수를 어떻게 합치는지, set을 이용한 중복 제거 트릭 등) 있으면 더 깊이 설명해드릴게요!

# 📦 1) Import 부분
# 💡 3가지 검색 기술이 등장합니다:

# 벡터 검색(similarity): 의미가 비슷한지 (임베딩 거리)
# BM25: 같은 단어가 들어있는지 (키워드 매칭)
# CrossEncoder: 질문-문서를 함께 보고 진짜 관련 있는지 정밀 채점 (재정렬)

from typing import List, Optional  # 타입 힌트 (List=리스트, Optional=값 또는 None)
from langsmith import traceable     # 함수 추적 데코레이터
from langchain.schema import Document  # 텍스트+메타데이터 표준 객체
from langchain.vectorstores.base import VectorStore  # 벡터DB의 기본 타입 (타입 힌트용)
from langchain_community.retrievers import BM25Retriever
# BM25: 키워드 기반 검색 알고리즘 (전통적인 검색엔진 방식)
from langchain.retrievers import EnsembleRetriever
# 여러 검색기를 가중치로 결합하는 앙상블 검색기
from sentence_transformers import CrossEncoder
# CrossEncoder: 질문-문서 쌍의 관련도를 정밀하게 점수화하는 모델 (재정렬용)

# 🎯 2) rerank_documents 함수 - 문서 재정렬 (re-ranking)
# 💡 왜 재정렬(re-rank)이 필요한가?
# 1차 검색(벡터/BM25)은 빠르지만 거칠어요. 수백 개 중 후보 10~20개를 빠르게 추리는 용도죠. CrossEncoder는 느리지만 정확해요. 그래서 "빠른 1차 검색 → 정밀한 2차 재정렬"의 2단계로 정확도를 높입니다.

# 🎯 2단계 검색 파이프라인 (이 프로젝트의 핵심 전략)
# [1단계: 빠른 후보 추출]          [2단계: 정밀 재정렬]
# 벡터 or 하이브리드 검색    →     CrossEncoder
# top_k개 (예: 20개) 추림          rerank_top_k개 (예: 5개)로 압축
# (빠르지만 거침)                  (느리지만 정확)

# 이렇게 하면 속도와 정확도를 둘 다 챙길 수 있어요. 전체 문서를 CrossEncoder로 다 채점하면 너무 느리니까, 1차로 후보를 좁힌 뒤 그것만 정밀 채점하는 거죠.

# 💡 Bi-Encoder vs Cross-Encoder 차이:
# 벡터 검색(Bi-Encoder): 질문과 문서를 따로 벡터화 후 거리 비교 (빠름, 덜 정확)
# CrossEncoder: 질문과 문서를 함께 모델에 넣어 채점 (느림, 정확)
@traceable(name="rerank_documents")  # 이 함수 추적
def rerank_documents(
    query: str,              # 사용자 질문
    docs: List[Document],    # 1차 검색으로 나온 문서들
    rerank_top_k: int,       # 재정렬 후 최종 반환할 개수
    verbose: bool            # 상세 로그 출력 여부
) -> List[Document]:
    """
    검색어와 문서 간 CrossEncoder 점수를 기반으로 문서를 재정렬하여 상위 N개를 반환합니다.
    """
    # 재정렬 전 기존 순서 출력 (디버깅용)
    if verbose:
        print("\n    📌 기존 문서 순서:")
        for i, doc in enumerate(docs, 1):  # 1번부터 번호 매김
            print(f"      {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}")

    # CrossEncoder 모델 로딩 (질문-문서 관련도를 정밀 채점하는 모델)
    # ms-marco-MiniLM: 검색 재정렬용으로 학습된 경량 모델
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 각 문서를 (질문, 문서본문) 쌍으로 묶음
    # → CrossEncoder는 이 쌍을 입력받아 관련도 점수를 매김
    pairs = [(query, doc.page_content) for doc in docs]
    
    # 모든 쌍에 대해 한 번에 점수 예측 (점수가 높을수록 관련도 높음)
    scores = model.predict(pairs)

    # 문서와 점수를 짝지음
    # zip: 두 리스트를 [(doc1, score1), (doc2, score2), ...] 형태로 묶음
    doc_scores = [(doc, score) for doc, score in zip(docs, scores)]
    
    # 점수 기준 내림차순 정렬 (관련도 높은 순)
    # key=lambda x: x[1] → 튜플의 두 번째 요소(점수)로 정렬
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # 정렬 결과 출력 (디버깅용)
    if verbose:
        # 상세 모드: 전체 문서 출력
        print("\n    📌 re-rank 적용 후 문서 순서:")
        for i, (doc, score) in enumerate(doc_scores, 1):
            print(f"      {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}, 점수: {score:.4f}")
    else:
        # 일반 모드: 상위 5개만 출력
        print("\n    📌 최종 문서 순서(상위 5개):")
        for i, (doc, score) in enumerate(doc_scores[:5], 1):
            print(f"      {i}. 파일명: {doc.metadata.get('파일명')}, 청크: {doc.metadata.get('chunk_idx')}, 점수: {score:.4f}")

    # 점수 높은 상위 rerank_top_k개의 "문서만" 추출하여 반환
    # [doc for doc, _ in ...] → 점수(_)는 버리고 문서만
    return [doc for doc, _ in doc_scores[:rerank_top_k]]

# 🔍 3) retrieve_documents 함수 - 검색 총괄 (메인)
# 🔄 전체 검색 흐름 정리
# retrieve_documents가 설정에 따라 분기하는 구조:
# retrieve_documents (검색 총괄)
#    │
#    ├─ search_type == "similarity"  →  벡터 검색만 (의미 기반)
#    │
#    ├─ search_type == "hybrid"      →  BM25(키워드 40%) + 벡터(의미 60%)
#    │                                   → 앙상블 결합 → 중복 제거
#    │
#    └─ rerank == True ?
#          ├─ Yes → rerank_documents (CrossEncoder 정밀 재정렬)
#          └─ No  → 상위 top_k개만

# 🧩 핵심 개념 3가지 비교
# 검색 방식 | 원리 | 장점 | 단점 |
# 벡터(similarity) 의미 유사도 동의어/문맥 이해 ("자동차"↔"차량")정확한 키워드 놓칠 수 있음
# BM25(키워드) 단어 빈도 매칭 고유명사/숫자에 강함 의미 이해 못함
# 하이브리드 둘을 가중 결합 두 장점 모두 더 무거움

# 💡 가중치 [0.4, 0.6]의 의미: BM25에 40%, 벡터에 60% 비중. RFP 문서는 의미 검색이 더 중요하다고 판단해 벡터에 더 높은 가중치를 줬어요. 이 숫자는 실험으로 튜닝하는 값입니다.

@traceable(name="retrieve_documents")  # 이 함수 추적
def retrieve_documents(
    vector_store: VectorStore,          # 벡터DB
    chunks: Optional[List[Document]],   # hybrid용 전체 청크 (없을 수도 있음)
    config: dict,                        # 설정
    chat_history: Optional[str] = None  # 대화 이력
) -> List[Document]:
    """
    주어진 쿼리에 대해 similarity 또는 hybrid 검색 방식으로 관련 문서를 검색합니다.
    """
    # ===== 설정값 꺼내기 =====
    query = config['retriever']['query']            # 사용자 질문
    search_type = config['retriever']['search_type'] # 검색 방식 (similarity/hybrid)
    top_k = config['retriever']['top_k']             # 검색할 문서 개수
    rerank = config['retriever']['rerank']           # 재정렬 사용 여부 (True/False)
    rerank_top_k = config['retriever']['rerank_top_k'] # 재정렬 후 최종 개수
    verbose = config['settings']['verbose']          # 상세 로그 여부

    # 대화 이력을 질문 앞에 붙여 맥락 보강
    # → 이전 대화를 반영해 검색 정확도 향상
    query = f"맥락: {chat_history}\n 질문:{query}"
    
    # ===== [방식 1] 유사도 검색 (similarity) =====
    if search_type == "similarity":
        # 벡터DB에서 질문과 의미적으로 가까운 top_k개 문서 검색
        docs = vector_store.similarity_search(query, k=top_k)
    
    # ===== [방식 2] 하이브리드 검색 (hybrid) =====
    elif search_type == "hybrid":
        # hybrid는 BM25에 전체 청크가 필요한데, 없으면 에러
        if chunks is None:
            raise ValueError("❌ [Value] (retrieval.retrieve_documents.chunks) chunks 누락 오류.")

        # --- 벡터 검색기 생성 ---
        try:
            # 벡터DB를 retriever(검색기) 형태로 변환
            vector_retriever = vector_store.as_retriever(
                search_type="similarity",      # 유사도 방식
                search_kwargs={"k": top_k}     # top_k개 반환
            )
        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (retrieval.retrieve_documents.vector_retriever) FAISS retriever 생성 실패: {e}")

        # --- BM25 키워드 검색기 생성 ---
        try:
            # 전체 청크로부터 BM25 검색기 생성 (키워드 빈도 기반)
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = top_k  # 반환 개수 설정
        except Exception as e:
            raise RuntimeError(f"❌ [Runtime] (retrieval.retrieve_documents.bm25_retriever) BM25 retriever 생성 실패: {e}")

        # --- 두 검색기를 앙상블로 결합 ---
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],  # 결합할 검색기들
            weights=[0.4, 0.6]  # 가중치: BM25 40%, 벡터검색 60% (벡터를 더 신뢰)
        )
        # 결합된 검색기로 검색 실행
        docs = hybrid_retriever.invoke(query)

        # --- 중복 제거 ---
        # 두 검색기가 같은 문서를 찾으면 중복될 수 있어 제거
        seen_pairs = set()    # 이미 본 문서를 기록하는 집합
        unique_docs = []
        for doc in docs:
            # (파일명, 청크번호)를 고유 식별자로 사용
            identifier = (doc.metadata.get("파일명"), doc.metadata.get("chunk_idx"))
            if identifier not in seen_pairs:  # 처음 보는 문서면
                unique_docs.append(doc)
                seen_pairs.add(identifier)    # 기록에 추가
        docs = unique_docs[:top_k]  # 중복 제거 후 top_k개만
    
    # ===== 지원하지 않는 방식 =====
    else:
        raise ValueError(f"❌ [Value] (retrieval.retrieve_documents.search_type) search_type 값 오류: {search_type}")
    
    # ===== 재정렬 적용 여부 =====
    if rerank:
        # CrossEncoder로 정밀 재정렬
        docs = rerank_documents(query, docs, rerank_top_k, verbose)
    else:
        # 재정렬 없이 상위 top_k개만
        docs = docs[:top_k]

    return docs  # 최종 검색 결과 반환