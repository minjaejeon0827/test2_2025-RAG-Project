# 소스코드 분석 (2026.05.28 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 드디어 사용자가 실제로 보는 UI 화면을 만드는 파일이네요! 지금까지 본 파일들이 "뒷단(엔진)"이었다면, 이 파일은 "앞단(운전석)"이에요. Streamlit으로 만든 챗봇 인터페이스이며, FastAPI 백엔드와 통신해 답변을 받아옵니다.이 파일은 길어서 9개 섹션으로 나눠 설명할게요.

# 🔄 전체 동작 흐름
# 사용자가 챗봇을 사용할 때 일어나는 일을 시간순으로 정리하면:
# 1. 브라우저 접속
#    └─ 페이지 헤더("RFPilot") 표시 + config.yaml 로딩

# 2. 사이드바에서 설정 조정 (선택)
#    └─ 슬라이더/드롭다운으로 config 딕셔너리 수정

# 3. 채팅창에 질문 입력
#    └─ st.chat_input("질문을 입력하세요")

# 4. 모델 로딩 (캐시되어 있으면 스킵)
#    └─ get_generation_model() → hf/openai_generator 호출

# 5. FastAPI 백엔드로 POST 요청 전송
#    └─ {질문, 대화이력, 세션ID, 설정} 전부 JSON으로
#    └─ 백엔드에서 RAG 파이프라인 실행 (loader → embedding → retrieval → generator)

# 6. 응답 받기
#    └─ {answer, elapsed, docs} 파싱

# 7. UI 업데이트
#    ├─ 탭1: 답변 + 추론 시간 표시, 대화 이력 갱신
#    └─ 탭2: 검색된 문서들을 expander로 표시

# 🎯 이 파일이 RAG 시스템에서 차지하는 역할
# 지금까지 본 모든 파일의 최종 합류 지점이에요:
# [프론트엔드 - chatbot.py]   ⬅️ 지금 여기
#         │ HTTP POST (질문 + 설정)
#         ▼
# [백엔드 - FastAPI 서버]
#         │
#         ├─ loader_main.py     (문서 준비)
#         ├─ embedding_main.py  (벡터DB)
#         ├─ retrieval_main.py  (검색)  
#         └─ hf/openai_generator.py (답변)
#         │
#         │ HTTP Response (답변 + 문서)
#         ▼
# [프론트엔드 - chatbot.py 화면 갱신]

# 이 아키텍처의 장점은 프론트와 백엔드 분리예요. 무거운 GPU 연산은 백엔드 서버에서, 가벼운 UI는 프론트에서 처리하니까 사용자 경험이 부드러워집니다. README에서 박규리님이 담당한 "FastAPI 연동 실험"(2025-05-22)이 바로 이 구조를 만든 거죠.
# 궁금한 부분(예: @st.cache_resource의 동작 원리, st.session_state의 수명주기, FastAPI 요청/응답 구조 등) 있으면 더 깊이 설명해드릴게요!
    

# 📦 1) Import 및 초기 설정
# 터미널 실행 코드
# python -m streamlit run src/streamlit/chatbot.py
# → 이 명령어로 Streamlit 앱을 실행

# ===== 외부 라이브러리 =====
import os         # 파일 경로/환경변수 처리
import shutil     # 디렉토리 삭제 등 파일 시스템 조작
import yaml       # config.yaml 파일 파싱/출력
import uuid       # 고유 식별자(세션 ID) 생성
import requests   # FastAPI 백엔드에 HTTP 요청 보내기
import torch      # PyTorch (GPU 관련 설정용)
import streamlit as st  # Streamlit UI 프레임워크
from typing import Dict  # 타입 힌트
from dotenv import load_dotenv  # .env 파일에서 환경변수 로딩

# ===== 프로젝트 내부 모듈 =====
from src.utils.config import load_config              # config.yaml 로딩
from src.utils.path import get_project_root_dir       # 프로젝트 루트 경로 얻기
from src.utils.shared_cache import set_cache_dirs     # 캐시 디렉토리 설정
from src.embedding.embedding_main import generate_index_name  # 벡터DB 인덱스 이름 생성
from src.generator.hf_generator import load_hf_model       # HuggingFace 모델 로더
from src.generator.openai_generator import load_openai_model  # OpenAI 모델 로더

set_cache_dirs()  # 캐시 폴더(HF/transformers 캐시 등) 경로 설정
load_dotenv()     # .env 파일에서 환경변수 읽어옴 (OPENAI_API_KEY 등)
torch.classes.__path__ = []
# Streamlit과 PyTorch 호환성 문제 해결용 워크어라운드 (경로 충돌 방지)
FASTAPI_URL = os.getenv("FASTAPI_URL")  # 백엔드 API 주소를 환경변수에서 가져옴

# 🎨 2) Streamlit 페이지 설정
# 💡 README의 "RFPilot" 브랜드 이름이 여기에 나오죠. 이 부분이 사용자가 처음 접하는 화면 맨 위 영역이에요.
# 브라우저 탭 제목/레이아웃 설정
st.set_page_config(
    page_title="2025-LLM-Project: RFP Summarizer & QA Chatbot", 
    layout="wide"  # 화면을 넓게 사용 (기본은 centered)
)

# 페이지 상단 헤더 (파란 구분선 포함)
st.header("RFPilot", divider='blue')
# 헤더 아래 작은 설명 글
st.caption("PDF, HWP 형식의 제안서를 기반으로 한 내용 요약 및 질의응답을 경험하세요!")

# ⚙️ 3) 프로젝트 설정 로딩
# 프로젝트 루트 경로와 config.yaml 로딩
try:
    project_root = get_project_root_dir()  # 프로젝트 최상위 폴더 경로
    config = load_config(project_root)      # config.yaml을 dict로 로딩
except Exception as e:
    # 실패하면 화면에 에러 표시하고 앱 중단
    st.error(f"❌ 설정 파일 로드 실패: {e}")
    st.stop()  # 아래 코드 실행 안 함
    
# ===== .env 파일 로딩 (API 키 등 비밀 정보) =====
dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)  # .env가 있으면 로딩
else:
    st.warning(".env 파일을 찾을 수 없습니다. 일부 기능이 제한될 수 있습니다.")
    
# 💾 4) 세션 상태 초기화 (대화 기억 장치)
# 💡 session_id는 여러 사용자가 동시 접속해도 각자의 벡터DB가 섞이지 않게 구분하는 역할이에요. FAISS 파일명에도 붙습니다({인덱스이름}_{세션ID}.faiss).
# st.session_state: 브라우저 새로고침해도 유지되는 데이터 저장소
# 대화창 같은 동적 UI에 필수

# ===== 대화 이력 초기화 =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 처음이면 빈 리스트로 시작
else:
    # 이미 있으면 유지 (이 분기는 사실상 같은 동작이라 중복이긴 함)
    st.session_state.chat_history = st.session_state.get("chat_history", [])
    config["chat_history"] = st.session_state.chat_history  # config에도 반영

# ===== 검색된 문서 저장소 초기화 =====
if "docs" not in st.session_state:
    st.session_state.docs = None  # 아직 검색 안 했으면 None

# ===== 세션 ID 생성 (각 사용자 구분용) =====
if "session_id" not in st.session_state:
    # uuid.uuid4(): 충돌 거의 없는 무작위 36자 ID 생성
    # 너무 길어서 앞 8자만 잘라서 사용
    st.session_state.session_id = str(uuid.uuid4())[:8]

session_id = st.session_state.session_id  # 편하게 쓰려고 변수에 저장

# 🤖 5) 모델 로딩 캐시 함수
# 💡 @st.cache_resource는 모델·DB 연결처럼 무거운 객체용이에요. 사용자가 채팅을 칠 때마다 Qwen 7B 모델을 다시 로딩하면 미친 짓이니까요.
# @st.cache_resource: Streamlit의 강력한 캐싱 데코레이터
# → 같은 인자로 호출하면 결과를 메모리에 저장해두고 재사용
# → 모델 로딩(수십 초~분)을 매번 반복하지 않음!
@st.cache_resource
def get_generation_model(model_type: str, model_name: str, use_quantization: bool = False) -> Dict:
    """
    지정된 모델 타입 및 이름에 따라 생성 모델을 로드합니다.
    """
    try:
        # 미니 config 딕셔너리 생성 (load_hf/openai_model이 요구하는 형식)
        config = {
            'generator': {
                'model_type': model_type,
                'model_name': model_name,
                'use_quantization': use_quantization
            }
        }

        # 모델 타입에 따라 분기 (앞서 분석한 두 함수 호출)
        if model_type == 'huggingface':
            return load_hf_model(config)  # HF 모델 로딩
        elif model_type == 'openai':
            return load_openai_model(config)  # OpenAI 클라이언트 생성
        else:
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        st.stop()
        
# 🔑 6) API 키 검증 함수
# ⚠️ 주의: 함수명이 api_key_verification(embed_model)인데 사이드바에서 model_type도 같은 함수에 전달됩니다. 이름과 실제 인자가 살짝 어긋나는 부분이 있어요.
def api_key_verification(embed_model):
    # 임베딩 모델이 'openai'인 경우에만 검사
    if embed_model.strip().lower() == "openai":
        load_dotenv()  # .env 다시 로드
        openai_key = os.environ.get("OPENAI_API_KEY")  # 환경변수에서 키 확인

        if not openai_key:
            # 키가 없으면 화면에 입력 폼 표시
            openai_key = st.text_input(
                "🔑 OpenAI API Key",
                type="password",  # 입력값을 ●●●●로 가림
                key=f"openai_api_key_special"  # 위젯 식별 키
            )
            if openai_key:
                # 입력받은 키를 환경변수에 설정 (이번 세션 동안 유효)
                os.environ["OPENAI_API_KEY"] = openai_key
            else:
                st.warning("OpenAI 모델을 사용하려면 API 키를 입력해야 합니다.")
                
# 🎛️ 7) 사이드바 (설정 패널)
# 💡 핵심 포인트: 사이드바의 모든 위젯이 config 딕셔너리를 직접 수정해요. 사용자가 슬라이더를 움직이면 즉시 config["retriever"]["top_k"] 같은 값이 바뀝니다. 이 config가 나중에 FastAPI로 통째로 전송됩니다.
# with st.sidebar: 이 블록 안의 위젯은 모두 왼쪽 사이드바에 배치
with st.sidebar:
    st.subheader("⚙️ 설정")
    
    # ===== 📂 데이터 설정 =====
    st.subheader("📂 데이터 설정")
    # 슬라이더: 최소~최대~기본값
    config["data"]["top_k"] = st.slider("🔢 최대 문서 수(files)", 1, 100, config["data"]["top_k"])
    
    # 드롭다운: 옵션 리스트와 현재 선택 인덱스
    config["data"]["file_type"] = st.selectbox(
        "📄 파일 유형", 
        ["all", "pdf", "hwp"], 
        index=["all", "pdf", "hwp"].index(config["data"]["file_type"])  # 현재값의 위치 찾기
    )
    
    # 토글 스위치 (on/off)
    config["data"]["apply_ocr"] = st.toggle("🧾 OCR 적용 여부", config["data"]["apply_ocr"])
    
    # 분할 방식 선택 (앞서 splitter.py에서 본 3가지!)
    config["data"]["splitter"] = st.selectbox(
        "✂️ 문서 분할 방법", 
        ["section", "recursive", "token"],
        index=["section", "recursive", "token"].index(config["data"]["splitter"])
    )
    
    # 숫자 입력 (chunk 크기/오버랩 - splitter.py로 전달됨)
    config["data"]["chunk_size"] = st.number_input("📏 Chunk 크기", value=config["data"]["chunk_size"], step=100)
    config["data"]["chunk_overlap"] = st.number_input("🔁 Chunk 오버랩", value=config["data"]["chunk_overlap"], step=10)

    # ===== 🧠 임베딩 설정 =====
    st.subheader("🧠 임베딩 설정")
    config["embedding"]["embed_model"] = st.text_input("🧬 임베딩 모델", config["embedding"]["embed_model"])
    config["embedding"]["db_type"] = st.selectbox(
        "💾 Vector DB 타입", 
        ["faiss", "chroma"], 
        index=["faiss", "chroma"].index(config["embedding"]["db_type"])
    )

    # 임베딩 모델이 openai면 API 키 입력 폼 표시
    api_key_verification(config["embedding"]["embed_model"])

    # ===== 🔍 리트리버 설정 (앞서 retrieval.py에서 본 옵션들!) =====
    st.subheader("🔍 리트리버 설정")
    config["retriever"]["search_type"] = st.selectbox(
        "🔎 검색 방식", 
        ["similarity", "hybrid"],  # similarity=벡터만 / hybrid=BM25+벡터
        index=["similarity", "hybrid"].index(config["retriever"]["search_type"])
    )
    config["retriever"]["top_k"] = st.slider("📄 검색 문서 수(chunks)", 1, 20, config["retriever"]["top_k"])
    config["retriever"]["rerank"] = st.toggle("📊 리랭크 적용", config["retriever"]["rerank"])  # CrossEncoder 사용 여부
    config["retriever"]["rerank_top_k"] = st.slider("🔝 리랭크 문서 수(chunks)", 1, 20, config["retriever"]["rerank_top_k"])

    # ===== 🔍 생성자 설정 =====
    ----- st.subheader("🔍 생성자 설정")
    ----- config["generator"]["model_type"] = st.selectbox(
        "🔎 생성 모델 타입", 
        ----- ["huggingface", "openai", "claude"],  # hf_generator / openai_generator / claude
        index=["huggingface", "openai", "claude"].index(config["generator"]["model_type"])
    )
    config["generator"]["model_name"] = st.text_input("🧬 생성 모델", config["generator"]["model_name"])
    config["generator"]["max_length"] = st.number_input("🔢 최대 토큰 수(max_length)", value=config["generator"]["max_length"], step=32)

    # 생성 모델이 openai면 API 키 한 번 더 확인
    api_key_verification(config["generator"]["model_type"])

    # 벡터DB 초기화 버튼 생성
    reset_vector_db = st.button("⚠️ Vector DB 초기화")
    
# 🗑️ 8) 벡터DB 초기화 로직
# ===== 벡터DB 파일 경로 결정 =====
    if config["embedding"]["db_type"] == "faiss":
        faiss_index_name = f"{generate_index_name(config)}"
        # FAISS는 .faiss(인덱스)와 .pkl(메타데이터) 두 파일로 저장됨
        vector_db_file = os.path.join(project_root, config['embedding']['vector_db_path'], f"{faiss_index_name}_{session_id}.faiss")
        metadata_file = os.path.join(project_root, config['embedding']['vector_db_path'], f"{faiss_index_name}_{session_id}.pkl")
    else:
        # Chroma는 폴더 단위로 저장됨
        chroma_folder_name = f"{generate_index_name(config)}_{session_id}"
        chroma_path = os.path.join(project_root, config['embedding']['vector_db_path'], chroma_folder_name)

    # ===== 초기화 버튼이 눌렸을 때 =====
    if reset_vector_db:
        try:
            if config["embedding"]["db_type"] == "faiss":
                if os.path.exists(vector_db_file):
                    os.remove(vector_db_file)   # .faiss 파일 삭제
                    os.remove(metadata_file)    # .pkl 파일 삭제
                    st.success("FAISS DB 삭제 완료")
                else:
                    st.info("FAISS 파일이 존재하지 않습니다.")
            elif config["embedding"]["db_type"] == "chroma":
                import shutil  # 이미 위에서 import 했지만 다시 import (큰 문제는 아님)
                if os.path.exists(chroma_path):
                    shutil.rmtree(chroma_path)  # 폴더 통째로 삭제
                    st.success("Chroma DB 삭제 완료")
                else:
                    st.info("Chroma 폴더가 존재하지 않습니다.")
        except Exception as e:
            st.error(f"Vector DB 삭제 실패: {e}")
            
    # ===== 리셋/모델 리로드 버튼 (2개 컬럼) =====
    cols = st.columns([4, 6])  # 4:6 비율로 두 컬럼
    with cols[0]:
        if st.button("🔄 리셋"):
            # 대화/문서 상태 초기화 (DB는 그대로)
            st.session_state.chat_history = []
            st.session_state.docs = None
            st.session_state.past_chunks = []
            st.rerun()  # 페이지 새로고침
    with cols[1]:
        if st.button("🔁 모델 리로드"):
            get_generation_model.clear()  # 캐시된 모델 메모리에서 제거
            st.rerun()
            
# 💬 9) 메인 컨텐츠 - 탭1: 챗봇
# 두 개의 탭 생성
tab1, tab2 = st.tabs(["💬 챗봇", "📄 문서 요약 및 분석"])

# 자주 쓸 설정값 변수에 저장
model_type = config["generator"]["model_type"]
model_name = config["generator"]["model_name"]
use_quantization = config["generator"]["use_quantization"]

# 디버깅용: 최종 config를 yaml 형식으로 콘솔에 출력
print("\n📄 [Verbose] 최종 설정 내용:")
print(yaml.dump(config, allow_unicode=True, sort_keys=False))

# ===== 탭1: 챗봇 =====
with tab1:
    # 화면 하단에 채팅 입력창 표시
    query = st.chat_input("질문을 입력하세요")

    # 사용자가 질문을 입력하고 엔터를 쳤을 때
    if query:
        # 입력값 검증 (빈 문자열 방지)
        if not isinstance(query, str) or query.strip() == "":
            st.warning("질문을 올바르게 입력해주세요.")
            st.stop()

        # ===== 벡터DB 저장/재사용 판단 =====
        # top_k가 100(=전체 문서)이면 매번 새로 만들지 않고 기존 DB 재사용
        if config["data"]["top_k"] == 100:
            if config["embedding"]["db_type"] == "faiss":
                is_save = not os.path.exists(vector_db_file)  # 없으면 새로 생성
            elif config["embedding"]["db_type"] == "chroma":
                is_save = not os.path.exists(chroma_path)
            else:
                is_save = True
        else:
            # top_k가 동적으로 바뀌면 항상 새로 만듦
            is_save = True
            
        # 새 질문이 들어오면 이전 검색 결과 초기화
        if st.session_state.docs is not None:
            st.session_state.docs = None
        
        # 모델 로딩 (캐시되어 있으면 즉시 반환)
        model_info = get_generation_model(model_type, 
                                        model_name, 
                                        use_quantization)
        
        # 아래 주석친 코드는 실행할 필요 없음(2026.05.28 minjae)
        # ⚠️ 버그! 아래 줄은 파이썬 문법 오류 (다섯 개의 하이픈 '-----' 때문)
        # 원래는 `chat_history = load_chat_history(config, model_info)`였을 텐데
        # 주석 처리하다 만 상태로 보임. 실행하면 SyntaxError 발생함
        # chat_history = load_chat_history(config, model_info)
        
        # 사용자 메시지를 채팅창에 표시
        with st.chat_message("user"):
            st.markdown(query)

        try:
            # 스피너(로딩 표시) + 답변 생성 요청
            with st.spinner("🤖 답변 생성 중..."):
                # FastAPI 백엔드로 POST 요청 전송
                # → 무거운 RAG 파이프라인은 백엔드 서버가 처리
                response = requests.post(
                    FASTAPI_URL,  # 백엔드 주소 (.env에서 로드)
                    json={
                        "query": query,                                  # 사용자 질문
                        "chat_history": st.session_state.chat_history,   # 이전 대화
                        "session_id": st.session_state.session_id,       # 세션 식별자
                        "config": config                                  # 사이드바 설정 전체
                    }
                )
                
                # HTTP 상태 코드 확인 (200 = 성공)
                if response.status_code != 200:
                    st.error(f"❌ API 요청 실패: {response.status_code} - {response.text}")
                    st.stop()
                    
                # 응답 JSON 파싱
                result = response.json()
                answer = result["answer"]    # 생성된 답변
                elapsed = result["elapsed"]  # 소요 시간
                docs = result["docs"]        # 참고한 문서들

            # 검색된 문서를 세션에 저장 (탭2에서 표시할 용도)
            st.session_state.docs = docs 
            
            # 대화 이력에 추가 (user/ai 쌍으로)
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "ai", "content": answer})
            
            # config에도 반영
            config["chat_history"] = st.session_state.chat_history
            
            # 추론 시간 표시
            with st.chat_message("assistant"):
                st.markdown(f"🕒 **추론 시간:** {elapsed}초")
                
            # 대화 이력이 너무 길면 최근 20개만 유지 (메모리/렌더링 부담 방지)
            MAX_CHAT_HISTORY = 20
            if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                # [-20:] → 마지막 20개만
                st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

        except Exception as e:
            st.error(f"❌ 문서 처리 중 오류 발생: {e}")
            st.stop()

    # ===== 기존 대화 이력 화면에 출력 =====
    # [::-1]: 리스트 역순 (최신 대화가 위에 오도록)
    for turn in st.session_state.chat_history[::-1]:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])
            
# 📄 10) 메인 컨텐츠 - 탭2: 문서 요약 및 분석
with tab2:
    st.subheader("📄 문서 요약 및 분석")

    # 세션에서 검색된 문서 가져오기 (없으면 None)
    docs = st.session_state.get("docs", None)

    if docs is None:
        # 아직 검색 안 했으면 안내 메시지
        st.info("❗ 먼저 질문을 입력하고 문서를 검색하세요.")
    elif isinstance(docs, list) and len(docs) > 0:
        # 문서 리스트가 있으면 각각 펼침 가능한 박스로 표시
        for i, doc in enumerate(docs):
            # st.expander: 클릭하면 펼쳐지는 접이식 UI
            with st.expander(f"[{i+1}] {doc['metadata'].get('사업명', '제목 없음')}"):
                st.write("📄 **메타데이터**")
                st.json(doc["metadata"])   # JSON 형태로 예쁘게 표시 (사업명/파일명 등)
                st.write("📝 **문서 내용**")
                st.write(doc["content"])    # 청크 본문 표시
    elif isinstance(docs, list) and len(docs) == 0:
        # 빈 리스트면 검색 결과 없음
        st.warning("검색된 문서가 없습니다.")
    else:
        # 그 외 (단일 객체인 경우)
        st.info(docs.page_content)
        
