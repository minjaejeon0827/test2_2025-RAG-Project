# 소스코드 분석 (2026.05.26 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 이 파일은 loader_main.py와 형제 같은 구조예요. 로더가 "문서 준비"를 총괄했다면, 이 파일은 "검색(retrieval)"을 총괄하는 지휘자입니다. 사용자 질문이 들어오면 벡터DB에서 관련 문서를 찾아오는 단계죠. README 구조도의 src/retriever/에 해당합니다.

# 📦 1) Import 부분
# 💡 임베딩 모델이 2종류(OpenAI/HuggingFace) 임포트된 이유는, 이 프로젝트가 두 가지를 모두 지원하기 때문이에요(README의 모델 라이센스 항목 참고: KoE5는 HF, text-embedding-3-large는 OpenAI).
from typing import List, Optional, Union
# List: 리스트 타입 힌트
# Optional: "이 값이거나 None일 수 있음" (예: Optional[str] = str 또는 None)
# Union: "이것 또는 저것 타입" (예: Union[A, B] = A 또는 B)

from langsmith import traceable  # 함수 전체를 추적하는 데코레이터
from langchain.schema import Document  # 텍스트+메타데이터를 담는 표준 객체

from langchain_openai import OpenAIEmbeddings
# OpenAI 임베딩 모델 (예: text-embedding-3-large)
from langchain_huggingface import HuggingFaceEmbeddings
# HuggingFace 임베딩 모델 (예: KoE5)

# ===== 프로젝트 내부 모듈 =====
from src.retrieval.retrieval import retrieve_documents
# 실제 검색을 수행하는 함수
from src.embedding.vector_db import load_vector_db
# 저장된 벡터DB를 디스크에서 불러오는 함수
from src.embedding.embedding_main import generate_index_name
# 설정값으로 벡터DB의 인덱스 이름을 만들어주는 함수

# 🔄 전체 흐름 요약
# retrieval_main은 단순하지만 중요한 2단계를 수행합니다:
# 단계함수/동작하는 일
# 1. DB 준비load_vector_db (조건부)벡터DB가 아직 메모리에 없으면 디스크에서 로딩
# 2. 검색 실행retrieve_documents질문과 유사한 문서 청크들을 벡터DB에서 검색

# 🧩 핵심 설계 포인트 3가지
# 1) if vector_store is None — 조건부 로딩
# 벡터DB 로딩은 무거운 작업이에요. 이미 불러온 적이 있으면 재사용하고, 없을 때만 새로 로딩합니다. 불필요한 중복 작업을 피하는 최적화죠.
# 2) Union[HuggingFaceEmbeddings, OpenAIEmbeddings] — 유연한 모델 지원
# 타입 힌트만 봐도 "이 프로젝트는 HF와 OpenAI 임베딩을 둘 다 받을 수 있다"는 걸 알 수 있어요. 설정만 바꾸면 모델 교체가 가능한 구조입니다.
# 3) similarity vs hybrid — 두 검색 방식
# docstring에 나온 두 방식의 차이:

# similarity(유사도) 검색: 질문과 의미가 비슷한 문서를 벡터 거리로 찾음 (의미 기반)
# hybrid(하이브리드) 검색: 의미 기반 + 키워드 기반(BM25 등)을 결합 → 그래서 chunks(전체 청크)가 필요함


# 💡 hybrid 검색에 chunks 전체가 필요한 이유: 키워드 검색은 벡터DB에 저장된 임베딩만으로는 부족하고, 원본 텍스트 전체를 훑어야 하기 때문이에요.

# 🎯 전체 RAG 파이프라인에서의 위치
# 지금까지 본 파일들을 연결하면 이런 큰 그림이 됩니다:
# main.py (전체 진입점)
#    │
#    ├─ loader_main.py      → 문서 로딩/전처리/청킹  ✅ (이전에 분석)
#    │     └─ data_loader.py, splitter.py
#    │
#    ├─ embedding_main.py   → 청크를 벡터로 변환 후 벡터DB 저장
#    │
#    ├─ retrieval_main.py   → 질문으로 관련 문서 검색  ⬅️ 지금 여기!
#    │     └─ retrieval.py, vector_db.py
#    │
#    └─ generator (hf/openai_generator.py) → 검색 문서 기반 답변 생성  ✅ (이전에 분석)

# 이제 RAG의 R(Retrieval, 검색) 부분 골격이 보이시죠? 이 함수가 찾아온 docs(관련 문서들)가 앞서 분석한 hf_generator.py/openai_generator.py의 프롬프트 안에 들어가서 최종 답변이 만들어지는 거예요.
# 다음에 retrieve_documents(실제 검색 로직)나 vector_db.py(DB 로딩)를 보여주시면, "유사도 검색이 내부적으로 어떻게 동작하는지"까지 완성해서 설명해드릴 수 있어요!

# 🎯 2) retrieval_main 함수
@traceable(name="retrieval_main")
# 이 함수 전체의 입력/출력/실행시간을 LangSmith에 자동 기록
def retrieval_main(
    config: dict,                        # 시스템 설정 딕셔너리
    vector_store: Optional[object],      # 이미 로드된 벡터DB (없으면 None)
    chunks: List[Document],              # hybrid 검색 시 쓸 전체 문서 청크
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],  # 임베딩 모델 (둘 중 하나)
    chat_history: Optional[str] = None   # 대화 이력 (없으면 None, 기본값)
) -> List[Document]:
    """
    설정에 따라 similarity 또는 hybrid 방식으로 검색하고, 필요시 re-ranking을 수행합니다.
    """
    # ===== 벡터DB 식별 정보 준비 =====
    # 설정값(모델명, 청크 크기 등)을 조합해 인덱스 이름 생성
    # → 어떤 벡터DB 파일을 불러올지 식별하는 키
    index_name = generate_index_name(config)

    # 벡터DB가 저장된 폴더 경로
    vector_db_path = config["embedding"]["vector_db_path"]
    
    # 벡터DB 종류 (예: "faiss" 또는 "chroma")
    db_type = config["embedding"]["db_type"]

    # ===== 벡터DB 로딩 (필요한 경우만) =====
    # vector_store가 None이면 (= 아직 안 불러왔으면) 디스크에서 로드
    # 이미 들어와 있으면 이 블록을 건너뛰어 중복 로딩 방지 (효율적)
    if vector_store is None:
        vector_store = load_vector_db(
            path=vector_db_path,      # DB 경로
            embeddings=embeddings,    # 임베딩 모델 (질문 벡터화에 필요)
            index_name=index_name,    # 어떤 인덱스를 쓸지
            db_type=db_type,          # FAISS/Chroma 등
        )

    # ===== 실제 검색 수행 =====
    # 핵심 함수 호출 - 질문과 관련된 문서들을 벡터DB에서 찾아옴
    docs = retrieve_documents(
        vector_store=vector_store,    # 검색할 벡터DB
        chunks=chunks,                # hybrid 검색용 전체 청크
        config=config,                # 검색 방식/개수 등 설정
        chat_history=chat_history     # 대화 맥락
    )
    print("✅ 문서 검색 완료")
    
    # ===== 상세 로그 출력 (디버깅용) =====
    verbose = config["settings"]["verbose"]  # 상세 출력 여부

    if verbose:
        count = 0
        # enumerate(docs, 1): 1번부터 번호 매기며 순회
        for i, doc in enumerate(docs, 1):
            print(f"\n    📄 문서 {i}")
            # 본문 앞 100자만 미리보기로 출력
            print(f"    - 본문(100자): {doc.page_content[:100]}...")
            # 메타데이터(사업명, 발주기관, 파일명 등) 출력
            print(f"    - 메타데이터: {doc.metadata}")
            count += 1
            if count > 4:  # 최대 5개(0~4)까지만 출력하고 멈춤
                break

    # 검색된 문서 리스트 반환 → 다음 단계인 generator(응답 생성)로 전달
    return docs
