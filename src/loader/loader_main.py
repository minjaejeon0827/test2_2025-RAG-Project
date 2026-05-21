# 소스코드 분석 (2026.05.21 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 이 파일은 앞서 본 data_loader.py의 함수들을 하나로 묶어 순서대로 실행시키는 "지휘자(orchestrator)" 역할이에요. README 구조도의 src/loader/에서 전체 로딩 흐름을 총괄하는 진입점입니다.

# 🧩 with trace(...)를 왜 단계마다 쓸까?
# 각 단계를 별도의 trace로 감싸면 LangSmith에서 "어느 단계가 얼마나 오래 걸렸는지", "어느 단계에서 에러가 났는지" 를 단계별로 시각화할 수 있어요. 예를 들어 PDF가 많아서 2단계(전처리)가 느리다면 그걸 바로 짚어낼 수 있죠.
# loader_main (@traceable)
# ├── load_data        ← 1단계 추적
# ├── process_data     ← 2단계 추적
# └── chunk_documents  ← 3단계 추적
# 🎯 RAG 파이프라인에서의 위치
# 이 함수는 "문서 준비 공장"의 컨베이어 벨트예요. main.py가 loader_main을 호출하면, 원본 파일들이 → 검색 → 추출 → 분할 → 검사를 거쳐 임베딩하기 좋은 청크 리스트로 가공되어 나옵니다. 이 청크들이 다음 단계인 src/embedding/으로 넘어가 벡터DB가 만들어지는 거죠.
# 다음에 splitter.py(data_chunking, summarize_chunk_quality의 실제 구현)를 보여주시면 청크가 구체적으로 어떻게 쪼개지는지까지 연결해서 설명해드릴 수 있어요!

# 📦 1) Import 부분
# 💡 Document 객체란? RAG에서 텍스트를 다루는 표준 단위예요. page_content(본문)와 metadata(출처/페이지 등 부가정보)를 함께 묶어 관리합니다.
from typing import List  # 타입 힌트용 - 반환값이 '리스트'임을 명시
from langsmith import trace, traceable
# trace: 코드 블록 단위로 실행을 추적/기록 (with 구문으로 사용)
# traceable: 함수 전체를 추적하는 데코레이터 (@traceable 형태로 사용)
from langchain.schema import Document
# LangChain의 Document 클래스 - 텍스트 내용 + 메타데이터를 담는 표준 객체

# ===== 우리 프로젝트 내부 모듈 불러오기 =====
from src.loader.data_loader import retrieve_top_documents_from_metadata, data_process
# 앞서 분석한 data_loader.py에서 두 함수 가져옴
# - retrieve_top_documents_from_metadata: 메타데이터 기반 유사 문서 검색
# - data_process: PDF/HWP 파일에서 텍스트 추출

from src.loader.splitter import data_chunking, summarize_chunk_quality
# splitter.py에서 두 함수 가져옴
# - data_chunking: 긴 텍스트를 작은 청크(조각)로 분할
# - summarize_chunk_quality: 생성된 청크의 품질 점검

# 🎯 2) loader_main 함수
@traceable(name="loader_main")
# 이 데코레이터는 loader_main 함수 전체의 입력/출력/실행시간을
# LangSmith에 자동 기록 → 디버깅과 성능 분석에 활용
def loader_main(config: dict, embeddings, chat_history) -> List[Document]:
    """
    설정 정보를 기반으로 문서를 로드하고, 전처리 및 청크 작업을 수행합니다.
    """
    # ===== config에서 필요한 설정값들을 미리 꺼내기 =====
    data_config = config["data"]              # data 관련 설정 묶음
    query = config["retriever"]["query"]      # 사용자의 검색 질문
    top_k = config["data"]["top_k"]           # 검색할 상위 문서 개수
    verbose = config["settings"]["verbose"]   # 상세 로그 출력 여부 (True/False)

    # ===== [1단계] 데이터 로드 (유사 문서 검색) =====
    # with trace(...): 이 블록의 실행을 "load_data"라는 이름으로 추적
    with trace(name="load_data"):
        # CSV 경로를 config에서 가져옴 (없으면 기본 경로 사용)
        data_list_path = data_config.get("data_list_path", "data/data_list.csv")
        
        # data_loader의 함수 호출 → 질문과 관련된 상위 top_k개 문서 메타데이터 반환
        df = retrieve_top_documents_from_metadata(
            query=query,                # 사용자 질문
            csv_path=data_list_path,    # 메타데이터 CSV 경로
            embeddings=embeddings,      # 임베딩 모델
            chat_history=chat_history,  # 대화 이력 (맥락 보강용)
            top_k=top_k,                # 상위 몇 개를 가져올지
        )
        print("✅ 문서 유사도 검색 완료")

    # ===== [2단계] 데이터 전처리 (실제 파일에서 텍스트 추출) =====
    with trace(name="process_data"):
        file_type = config["data"]["file_type"]  # 처리할 파일 종류 (hwp/pdf/all)
        apply_ocr = config["data"]["apply_ocr"]   # OCR 적용 여부
        
        # 1단계에서 추린 문서들의 실제 파일을 열어 본문 텍스트를 추출
        # → df에 "full_text" 컬럼이 추가됨
        df = data_process(df, config=config, apply_ocr=apply_ocr, file_type=file_type)
        print("✅ 데이터 전처리 완료")

    # ===== [3단계] 청크 생성 (긴 텍스트를 작은 조각으로 분할) =====
    with trace(name="chunk_documents"):
        splitter_type = config["data"]["splitter"]       # 분할 방식 (예: recursive 등)
        chunk_size = config["data"]["chunk_size"]         # 한 청크의 최대 길이
        chunk_overlap = config["data"]["chunk_overlap"]   # 청크 간 겹치는 길이

        # 본문 텍스트를 검색하기 좋은 작은 단위로 쪼갬
        # → Document 객체들의 리스트 반환
        chunks = data_chunking(
            df=df,                       # full_text가 담긴 DataFrame
            splitter_type=splitter_type, # 분할 알고리즘
            size=chunk_size,             # 청크 크기
            overlap=chunk_overlap,       # 겹침 크기
        )
        print("✅ 청크 생성 완료")

    # ===== [4단계] 청크 품질 검사 =====
    # 생성된 청크들의 통계(개수, 평균 길이 등)를 점검
    summarize_chunk_quality(chunks, verbose)
    if verbose:  # 상세 로그 모드일 때만 완료 메시지 출력
        print("✅ 청크 품질 검사 완료")

    # 최종 결과: 청크 리스트 반환 → 이후 임베딩/벡터DB 단계로 전달됨
    return chunks