# 소스코드 분석 (2026.05.12 minjae)
# 참고: https://gemini.google.com/app/7192578f5d43d984?hl=ko

# 안녕하세요! 첨부해주신 소중한 프로젝트 설명서(README)와 코드를 꼼꼼하게 모두 읽어보았습니다. 매일 수백 건씩 쏟아지는 제안요청서(RFP)를 인공지능으로 빠르게 요약하고 질문에 답해주는 'RFPilot' 프로젝트라니, 정말 멋지고 실용적인 아이디어네요!

# 작성해주신 chatbot_original.py 파일은 사용자가 인터넷 브라우저를 통해 직접 챗봇과 대화하고 설정을 바꿀 수 있도록 화면(UI)을 만들어주는 아주 중요한 '스트림릿(Streamlit)' 코드입니다.

# 코드가 길고 복잡해서 처음엔 낯설게 느껴지는 게 당연해요. 중학생 친구에게 설명하듯, 어려운 컴퓨터 용어는 최대한 빼고 파일의 처음부터 끝까지 코드 한 줄 한 줄의 의미를 친절하게 주석으로 달아드릴게요. 천천히 따라와 보세요!

# 어떠신가요? 코드가 마치 집을 짓거나 공장을 돌리는 과정처럼 하나하나 순서대로 맞물려 돌아가고 있어요! 크게 '준비물 챙기기 ➡️ 화면 디자인 ➡️ 설정 메뉴 띄우기 ➡️ 질문 받고 답변 내놓기' 순서로 작동한다고 생각하면 훨씬 이해하기 편할 거예요.

# 이 프로젝트가 현장에서 컨설턴트 분들의 소중한 시간을 정말 많이 아껴줄 수 있을 것 같아 기대가 큽니다. 더 궁금한 부분이나 이해가 안 가는 줄이 있다면 언제든 편하게 질문해주세요! 😊

# 터미널에서 이 화면을 실행하기 위한 명령어예요. (명령프롬프트에 입력하는 용도)
# python -m streamlit run src/streamlit/chatbot.py

# ==========================================
# 1. 외부 도구(라이브러리) 가져오기
# ==========================================
import os              # 컴퓨터의 폴더나 파일 경로를 다루기 위한 도구예요.
import shutil          # 파일을 복사하거나 통째로 지울 때 쓰는 청소부 같은 도구예요.
import yaml            # 설정 파일(.yaml)을 읽고 쓰기 위한 번역기예요.
import uuid            # 겹치지 않는 무작위의 고유한 이름(ID)을 만들어주는 도구예요.
import torch           # 인공지능 모델을 돌리기 위한 강력한 수학 계산 엔진이에요.
import streamlit as st # 우리가 웹사이트 화면(UI)을 아주 쉽게 만들 수 있게 해주는 핵심 마법 도구예요! (줄여서 st라고 부를게요)
from typing import Dict # 사전(Dictionary) 형태의 데이터라고 컴퓨터에게 힌트를 줄 때 써요.
from dotenv import load_dotenv # 비밀번호(API 키 등)가 담긴 숨김 파일(.env)을 읽어오는 도구예요.

# ==========================================
# 2. 우리 프로젝트 내부 도구 가져오기
# ==========================================
from src.utils.config import load_config                # 설정 파일을 읽어오는 우리 팀의 함수예요.
from src.utils.path import get_project_root_dir         # 우리 프로젝트 폴더의 맨 꼭대기(뿌리) 위치를 찾아줘요.
from src.utils.shared_cache import set_cache_dirs       # 인공지능 모델이 다운로드될 임시 저장소 위치를 정해줘요.
from src.embedding.embedding_main import generate_index_name # 문서들을 모아둔 책꽂이(DB)의 이름을 지어주는 함수예요.
from src.generator.hf_generator import load_hf_model    # 허깅페이스(무료 인공지능 창고)의 모델을 불러와요.
from src.generator.openai_generator import load_openai_model # 오픈AI(ChatGPT 등)의 똑똑한 모델을 불러와요.
from src.embedding.vector_db import generate_embedding  # 글자를 컴퓨터가 이해하는 숫자(벡터)로 바꿔주는 함수예요.
from src.generator.chat_history import load_chat_history # 챗봇이 예전에 했던 대화를 기억하게 불러오는 함수예요.

import sys
# 파이썬이 우리 프로젝트의 다른 폴더에 있는 코드도 찾을 수 있도록 길을 뚫어주는 작업이에요.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipe_main import rag_pipeline # 질문을 받고 -> 문서를 찾고 -> 답변을 만들어주는 핵심 공장(파이프라인)이에요!

# ==========================================
# 3. 기본 설정 및 초기화
# ==========================================
set_cache_dirs() # 모델 다운로드 폴더 설정 실행!
load_dotenv()    # 숨겨둔 비밀번호 파일(.env) 읽기 실행!
torch.classes.__path__ = [] # 파이토치(인공지능 엔진)가 엉뚱한 길로 새지 않게 경로를 비워둬요.
FASTAPI_URL = os.getenv("FASTAPI_URL") # 백엔드 서버(FastAPI)와 통신할 주소를 가져와요.

# ==========================================
# 4. 웹페이지 화면 디자인 (Streamlit)
# ==========================================
st.set_page_config(
    page_title="2025-LLM-Project: RFP Summarizer & QA Chatbot", # 인터넷 브라우저 맨 위 탭에 보일 이름이에요.
    layout="wide" # 화면을 좁게 쓰지 않고 모니터 너비에 맞춰 넓게 쓰겠다는 뜻이에요.
)

st.header("RFPilot", divider='blue') # 화면 맨 위에 파란색 밑줄과 함께 'RFPilot'이라는 큰 제목을 달아줘요.
st.caption("PDF, HWP 형식의 제안서를 기반으로 한 내용 요약 및 질의응답을 경험하세요!") # 제목 밑에 작은 글씨로 설명을 적어요.

# ==========================================
# 5. 설정 파일(Config) 로딩
# ==========================================
try:
    project_root = get_project_root_dir()     # 프로젝트의 가장 위쪽 폴더 위치를 찾아요.
    config = load_config(project_root)        # 그 위치에 있는 설정 파일을 읽어와요.
except Exception as e:
    # 만약 설정 파일을 못 찾거나 오류가 나면 ❌ 표시와 함께 에러를 띄우고 멈춰요!
    st.error(f"❌ 설정 파일 로드 실패: {e}")
    st.stop()

# .env 파일 로딩 (API Key 등 남에게 보여주면 안 되는 비밀 정보 처리용)
dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path): # 파일이 있으면
    load_dotenv(dotenv_path=dotenv_path) # 읽어오고
else:
    # 없으면 노란색 경고창을 띄워줘요.
    st.warning(".env 파일을 찾을 수 없습니다. 일부 기능이 제한될 수 있습니다.")

# ==========================================
# 6. 기억력(Session State) 초기화
# ==========================================
# 웹사이트는 새로고침하면 데이터를 까먹기 때문에 'session_state'라는 마법의 수첩에 중요한 걸 적어둬야 해요.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # 대화 기록 수첩이 없으면 새 수첩(빈 리스트)을 만들어요.
else: 
    # 수첩이 이미 있다면 그 수첩을 그대로 쓰고, 설정 파일에도 알려줘요.
    st.session_state.chat_history = st.session_state.get("chat_history", [])
    config["chat_history"] = st.session_state.chat_history

if "docs" not in st.session_state:
    st.session_state.docs = None # 찾은 제안서 문서들을 저장할 공간을 비워둬요.

if "session_id" not in st.session_state:
    # 이번 대화를 위한 고유한 이름표(ID)를 만들어요. 너무 길면 보기 싫으니까 앞의 8글자만 잘라서 쓸게요.
    st.session_state.session_id = str(uuid.uuid4())[:8]

session_id = st.session_state.session_id # 편하게 쓰기 위해 변수에 담아둬요.

# ==========================================
# 7. 핵심 함수 정의
# ==========================================

# @st.cache_resource는 "이 함수가 가져온 거대한 인공지능 뇌는 메모리에 잘 기억해둬!"라는 뜻이에요.
# 그래야 대화할 때마다 무거운 모델을 새로 불러오지 않아서 답변이 빨라집니다.
@st.cache_resource
def get_generation_model(model_type: str, model_name: str, use_quantization: bool = False) -> Dict:
    """
    지정된 모델 타입 및 이름에 따라 똑똑한 생성 모델(뇌)을 불러옵니다.
    """
    try:
        # 모델을 부르기 위한 설정값을 만들어요.
        config = {
            'generator': {
                'model_type': model_type,
                'model_name': model_name,
                'use_quantization': use_quantization # 양자화(모델의 덩치를 줄여 가볍게 만드는 기술) 사용 여부예요.
            }
        }

        if model_type == 'huggingface':
            return load_hf_model(config)     # 허깅페이스 무료 모델 부르기!
        elif model_type == 'openai':
            return load_openai_model(config) # 똑똑한 유료 모델 부르기!
        else:
            # 알 수 없는 모델이면 에러를 냅니다.
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    
    except Exception as e:
        # 에러가 나면 빨간 창으로 알려주고 멈춰요.
        st.error(f"모델 로딩 실패: {e}")
        st.stop()


def api_key_verification(embed_model):
    """
    OpenAI 같은 유료 모델을 쓸 때 입장권(API Key)이 있는지 검사하는 함수예요.
    """
    if embed_model.strip().lower() == "openai": # 만약 모델 이름에 openai가 들어가면
        load_dotenv()
        openai_key = os.environ.get("OPENAI_API_KEY") # 컴퓨터 설정에서 키를 찾아봐요.

        if not openai_key: # 키가 없다면?
            # 사용자에게 비밀번호 입력창을 띄워서 직접 키를 적으라고 해요.
            openai_key = st.text_input(
                "🔑 OpenAI API Key",
                type="password", # 입력할 때 별표(***)로 가려지게 해요.
                key=f"openai_api_key_special"
            )
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key # 입력한 키를 저장해요.
            else:
                st.warning("OpenAI 모델을 사용하려면 API 키를 입력해야 합니다.")

# ==========================================
# 8. 화면 왼쪽 메뉴바 (사이드바) 만들기
# ==========================================
with st.sidebar: # 이제부터 나오는 것들은 모두 왼쪽 사이드바에 들어갑니다!
    st.subheader("⚙️ 설정") # 작은 제목이에요.
    
    # [1] 데이터 관련 설정칸
    st.subheader("📂 데이터 설정")
    # 문서를 최대 몇 개까지 읽을지 슬라이더로 조절하게 해요. (1~100개)
    config["data"]["top_k"] = st.slider("🔢 최대 문서 수(files)", 1, 100, config["data"]["top_k"])
    # 문서의 종류를 선택하게 해요. (전체, PDF, HWP)
    config["data"]["file_type"] = st.selectbox("📄 파일 유형", ["all", "pdf", "hwp"], index=["all", "pdf", "hwp"].index(config["data"]["file_type"]))
    # 그림으로 된 글자도 읽을지(OCR) 스위치(토글)로 껐다 켤 수 있어요.
    config["data"]["apply_ocr"] = st.toggle("🧾 OCR 적용 여부", config["data"]["apply_ocr"])
    # 거대한 문서를 어떻게 자를지 방법을 선택해요.
    config["data"]["splitter"] = st.selectbox("✂️ 문서 분할 방법", ["section", "recursive", "token"], index=["section", "recursive", "token"].index(config["data"]["splitter"]))
    # 자를 때 한 덩어리(Chunk)의 크기를 정해요.
    config["data"]["chunk_size"] = st.number_input("📏 Chunk 크기", value=config["data"]["chunk_size"], step=100)
    # 덩어리와 덩어리 사이에 내용이 조금 겹치게 해서 맥락을 유지할 크기를 정해요.
    config["data"]["chunk_overlap"] = st.number_input("🔁 Chunk 오버랩", value=config["data"]["chunk_overlap"], step=10)

    # [2] 임베딩(글자를 숫자로 바꾸는) 설정칸
    st.subheader("🧠 임베딩 설정")
    config["embedding"]["embed_model"] = st.text_input("🧬 임베딩 모델", config["embedding"]["embed_model"])
    config["embedding"]["db_type"] = st.selectbox("💾 Vector DB 타입", ["faiss", "chroma"], index=["faiss", "chroma"].index(config["embedding"]["db_type"]))

    # 임베딩 모델에 OpenAI를 쓴다면 입장권이 있는지 방금 만든 함수로 확인해요.
    api_key_verification(config["embedding"]["embed_model"])

    # [3] 리트리버(질문에 맞는 문서를 찾는 탐정) 설정칸
    st.subheader("🔍 리트리버 설정")
    config["retriever"]["search_type"] = st.selectbox("🔎 검색 방식", ["similarity", "hybrid"], index=["similarity", "hybrid"].index(config["retriever"]["search_type"]))
    config["retriever"]["top_k"] = st.slider("📄 검색 문서 수(chunks)", 1, 20, config["retriever"]["top_k"])
    config["retriever"]["rerank"] = st.toggle("📊 리랭크 적용", config["retriever"]["rerank"]) # 찾은 문서를 다시 한번 꼼꼼히 점수를 매겨 줄 세울지(리랭크) 정해요.
    config["retriever"]["rerank_top_k"] = st.slider("🔝 리랭크 문서 수(chunks)", 1, 20, config["retriever"]["rerank_top_k"])

    # [4] 생성자(답변을 글짓기 하는 뇌) 설정칸
    st.subheader("🔍 생성자 설정")
    config["generator"]["model_type"] = st.selectbox("🔎 생성 모델 타입", ["huggingface", "openai"], index=["huggingface", "openai"].index(config["generator"]["model_type"]))
    config["generator"]["model_name"] = st.text_input("🧬 생성 모델", config["generator"]["model_name"])
    config["generator"]["max_length"] = st.number_input("🔢 최대 토큰 수(max_length)", value=config["generator"]["max_length"], step=32)

    # 생성 모델에 OpenAI를 쓴다면 입장권이 있는지 다시 확인해요.
    api_key_verification(config["generator"]["model_type"])

    # 벡터 데이터베이스(문서 책꽂이)를 비우는 버튼을 만들어요.
    reset_vector_db = st.button("⚠️ Vector DB 초기화")
    
    # 책꽂이 파일이 어디에 저장되어 있는지 경로를 미리 계산해둬요.
    if config["embedding"]["db_type"] == "faiss":
        faiss_index_name = f"{generate_index_name(config)}"
        vector_db_file = os.path.join(project_root, config['embedding']['vector_db_path'], f"{faiss_index_name}_{session_id}.faiss")
        metadata_file = os.path.join(project_root, config['embedding']['vector_db_path'], f"{faiss_index_name}_{session_id}.pkl")
    else:
        chroma_folder_name = f"{generate_index_name(config)}_{session_id}"
        chroma_path = os.path.join(project_root, config['embedding']['vector_db_path'], chroma_folder_name)


    if reset_vector_db: # 만약 초기화 버튼이 눌렸다면?
        try:
            # 설정된 타입에 따라 책꽂이 파일이나 폴더를 영구적으로 지워버려요.
            if config["embedding"]["db_type"] == "faiss":
                if os.path.exists(vector_db_file):
                    os.remove(vector_db_file)
                    os.remove(metadata_file)
                    st.success("FAISS DB 삭제 완료") # 성공했다고 초록색 알림을 줘요.
                else:
                    st.info("FAISS 파일이 존재하지 않습니다.")
            elif config["embedding"]["db_type"] == "chroma":
                import shutil
                if os.path.exists(chroma_path):
                    shutil.rmtree(chroma_path) # 폴더 안의 내용까지 싹 지워주는 청소부 도구예요.
                    st.success("Chroma DB 삭제 완료")
                else:
                    st.info("Chroma 폴더가 존재하지 않습니다.")
        except Exception as e:
            st.error(f"Vector DB 삭제 실패: {e}")
            
    # 기타 유용한 버튼들을 가로로 나란히 배치해요.
    cols = st.columns([4, 6]) # 비율을 4:6으로 쪼갠 2칸을 만들어요.
    with cols[0]:
        if st.button("🔄 리셋"): # 첫 번째 칸에 대화 내용을 싹 비우는 리셋 버튼!
            st.session_state.chat_history = []
            st.session_state.docs = None
            st.session_state.past_chunks = []
            st.rerun() # 웹페이지를 새로고침해요.
    with cols[1]:
        if st.button("🔁 모델 리로드"): # 두 번째 칸에 모델을 처음부터 다시 불러오는 버튼!
            get_generation_model.clear() # 기억해둔 모델 메모리를 강제로 지워요.
            st.rerun()

# ==========================================
# 9. 화면 가운데 탭(Tab) 만들기 (챗봇 vs 문서 요약)
# ==========================================
tab1, tab2 = st.tabs(["💬 챗봇", "📄 문서 요약 및 분석"])

# 편하게 쓰기 위해 변수에 빼둬요.
model_type = config["generator"]["model_type"]
model_name = config["generator"]["model_name"]
use_quantization = config["generator"]["use_quantization"]

# 터미널 창에 개발자가 볼 수 있게 최종 설정값들을 출력해줘요.
print("\n📄 [Verbose] 최종 설정 내용:")
print(yaml.dump(config, allow_unicode=True, sort_keys=False))

# [첫 번째 탭] 💬 챗봇 영역
with tab1:
    query = st.chat_input("질문을 입력하세요") # 맨 아래쪽에 카카오톡처럼 채팅을 칠 수 있는 입력창을 만들어요.

    if query: # 사용자가 질문을 치고 엔터를 쳤다면!
        if not isinstance(query, str) or query.strip() == "":
            # 이상한 글자거나 빈칸만 쳤으면 경고를 줘요.
            st.warning("질문을 올바르게 입력해주세요.")
            st.stop()

        # 새로 문서를 저장해야 할지(DB를 구워야 할지) 검사해요. 
        if config["data"]["top_k"] == 100:
            if config["embedding"]["db_type"] == "faiss":
                is_save = not os.path.exists(vector_db_file) # 파일이 없으면 저장해야지!(True)
            elif config["embedding"]["db_type"] == "chroma":
                is_save = not os.path.exists(chroma_path)
            else:
                is_save = True
        else:
            is_save = True
            
        # 새 질문이 들어왔으니 옛날에 띄워둔 문서는 일단 숨겨요.
        if st.session_state.docs is not None:
            st.session_state.docs = None
        
        # 아까 만든 함수로 똑똑한 인공지능 뇌를 불러와요!
        model_info = get_generation_model(model_type, model_name, use_quantization)

        # 챗봇이 그동안 사용자와 무슨 대화를 나눴는지 기억을 꺼내와요.
        chat_history = load_chat_history(config, model_info)
        
        # 화면에 내가 한 질문을 카카오톡 내 말풍선처럼 보여줘요.
        with st.chat_message("user"):
            st.markdown(query)

        try:
            with st.spinner("🤖 답변 생성 중..."): # 뺑글뺑글 돌아가는 로딩 화면을 보여줘요.
                # 글자를 숫자로 바꾸는 도구를 준비해요.
                embeddings = generate_embedding(config["embedding"]["embed_model"])
                
                # 대망의 핵심 공장! 파이프라인에 문서를 주고, 질문을 주고, 뇌를 줘서 최종 답변(answer)을 받아내요!
                docs, answer, elapsed = rag_pipeline(config, embeddings, chat_history, model_info=model_info, is_save=is_save, session_id=session_id)
                
            # 찾은 문서를 수첩에 기록해요.
            st.session_state.docs = docs 
            
            # 수첩에 방금 내가 한 질문과 인공지능이 한 대답을 차곡차곡 적어둬요.
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "ai", "content": answer})
            
            # 설정 파일에도 업데이트 해줘요.
            config["chat_history"] = st.session_state.chat_history
            
            # 인공지능 말풍선을 화면에 띄워요.
            with st.chat_message("assistant"):
                # 대답을 내놓는데 걸린 시간을 초 단위로 알려줘요.
                st.markdown(f"🕒 **추론 시간:** {elapsed}초")
                
            # 대화가 너무 많아지면 화면이 멈출 수 있으니, 최신 대화 20개만 남기고 옛날 대화는 지워요.
            MAX_CHAT_HISTORY = 20
            if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

        except Exception as e:
            # 문서 처리하다 펑! 하고 에러가 나면 멈추고 알려줘요.
            st.error(f"❌ 문서 처리 중 오류 발생: {e}")
            st.stop()

    # 스크롤을 내리거나 새로고침 해도 옛날 대화가 지워지지 않게 화면에 다시 쫙 뿌려주는 반복문이에요.
    for turn in st.session_state.chat_history[::-1]:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])


# [두 번째 탭] 📄 문서 요약 및 분석 영역
with tab2:
    st.subheader("📄 문서 요약 및 분석") # 탭의 큰 제목이에요.

    # 수첩(session_state)에 저장된 참고 문서가 있는지 꺼내봐요.
    docs = st.session_state.get("docs", None)

    if docs is None: # 아직 질문을 안 해서 문서가 없다면?
        st.info("❗ 먼저 질문을 입력하고 문서를 검색하세요.")
    elif isinstance(docs, list) and len(docs) > 0: # 찾은 문서들이 1개라도 무사히 리스트에 담겨 있다면?
        for i, doc in enumerate(docs):
            # 아코디언처럼 접었다 펼 수 있는(expander) 상자를 만들어서 문서 내용을 깔끔하게 정리해 보여줘요.
            with st.expander(f"[{i+1}] {doc['metadata'].get('사업명', '제목 없음')}"):
                st.write("📄 **메타데이터**")
                st.json(doc["metadata"]) # 작성일자, 페이지 번호 같은 부가 정보를 보기 좋게 출력해요.
                st.write("📝 **문서 내용**")
                st.write(doc["content"]) # 실제 제안서의 내용 텍스트를 출력해요.
    elif isinstance(docs, list) and len(docs) == 0:
        # 질문은 했는데 관련된 문서를 못 찾았을 때예요.
        st.warning("검색된 문서가 없습니다.")
    else:
        # 혹시 몰라 데이터 형태가 리스트가 아닐 때를 대비한 안전 장치예요.
        st.info(docs.page_content)