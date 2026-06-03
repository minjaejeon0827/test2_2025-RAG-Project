# 소스코드 분석 (2026.05.13 minjae)
# 참고: https://gemini.google.com/app/7192578f5d43d984?hl=ko

# 안녕하세요! 올려주신 RFPilot 프로젝트의 README 파일을 꼼꼼하게 처음부터 끝까지 다 읽어보았습니다.

# 이 프로젝트는 매일 쏟아지는 수백 건의 공공기관 제안요청서(RFP)를 인공지능이 대신 읽고 요약해 주는 아주 훌륭하고 실용적인 B2G 컨설팅 지원 도구네요! 
# 이전 코드(chatbot_original.py)가 사용자와 직접 만나는 '웹사이트 화면(프론트엔드)'이었다면, 
# 이번에 보여주신 main.py 코드는 그 화면 뒤에서 실제로 질문을 받고 묵묵히 대답을 만들어내는 '서버(백엔드)'를 만드는 코드입니다.

# 화면을 만드는 코드와는 또 다른 개념들이 나와서 헷갈리실 수 있어요. 이번에도 중학생 친구에게 설명하듯 어려운 전문 용어는 최대한 빼고, 아주 쉽고 친절하게 한 줄 한 줄 주석을 달아드릴게요. 편안하게 따라와 주세요! 😊

# 💻 main.py (FastAPI 백엔드 서버) 코드 상세 분석
# 이 코드는 쉽게 말해 "주문을 받는 식당의 주방(서버)"을 차리는 과정이라고 생각하시면 됩니다. 사용자가 웹사이트에서 질문(주문)을 던지면, 이 주방에서 데이터를 요리해 답변(음식)을 내어줍니다.

# 요약하자면!
# 이 파일은 "외부에서 질문을 받기 위한 대기 창구를 열고 ➡️ 질문이 들어오면 대화 이력을 합쳐서 ➡️ RAG 공장(파이프라인)으로 보내 답변을 만들고 ➡️ 만들어진 답변과 참고 문서를 다시 내보내주는 역할"을 완벽하게 수행하고 있습니다.

# 이전의 chatbot_original.py 파일이 여기서 만들어진 창구(/chat)로 질문을 휙 던지면, 이 서버가 답을 만들어서 다시 던져주는 방식으로 두 프로그램이 핑퐁 게임처럼 소통하는 구조랍니다.

# 팀원 분들과 함께 정말 탄탄한 시스템을 구축하셨네요! 코드를 보면서 이해가 안 가는 부분이 있다면 한 번 더 천천히 읽어보시고, 언제든 다시 찾아주세요. 응원합니다!

# ==========================================
# 1. 외부 도구(라이브러리) 가져오기
# ==========================================
import os # 컴퓨터 안의 폴더나 파일 위치를 찾아주는 길잡이예요.
from fastapi import FastAPI # 웹 서버(주방)를 아주 빠르고 쉽게 만들어주는 메인 요리사 도구예요.
from fastapi import HTTPException  # HTTP 통신 예외(Exception) 처리 클래스
from pydantic import BaseModel # 데이터가 규칙에 맞게 잘 들어왔는지 검사하는 깐깐한 '주문서 양식' 도구예요.
from typing import Optional # "이 항목은 적어도 되고 안 적어도 돼(선택사항이야)"라고 알려주는 힌트예요.
from fastapi.middleware.cors import CORSMiddleware # 다른 주소의 웹사이트에서도 우리 서버에 말을 걸 수 있도록 허락해 주는 '문지기'예요.
from copy import deepcopy # 원본 데이터를 망가뜨리지 않고 똑같이 복사본을 만들어주는 복사기예요.

# ==========================================
# 2. 우리 프로젝트 내부 도구 가져오기
# ==========================================
from src.generator.chat_history import load_chat_history # 과거 대화 내용을 불러오는 함수예요.
from src.utils.config import load_config                 # 설정 파일(config.yaml)을 읽어오는 함수예요.
from src.utils.path import get_project_root_dir          # 프로젝트의 맨 꼭대기(뿌리) 폴더 위치를 찾아줘요.
from src.embedding.vector_db import generate_embedding   # 글자를 컴퓨터가 이해하는 숫자(벡터)로 바꿔주는 함수예요.
from main import get_generation_model, rag_pipeline      # 인공지능 뇌를 불러오고, 최종 답변을 만들어내는 핵심 공장(파이프라인)을 가져와요.
from dotenv import load_dotenv                           # API 키 같은 비밀번호가 적힌 숨김 파일(.env)을 읽어오는 도구예요.

# ==========================================
# 3. 서버(주방) 오픈 준비
# ==========================================
app = FastAPI() # "이제부터 FastAPI 서버를 엽니다!" 하고 선언하는 거예요. (이름을 app이라고 지었어요)

# CORS 설정: 우리 식당(서버)에 누가 들어올 수 있는지 규칙을 정해요.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ["*"]는 "어떤 웹사이트에서 요청이 오든 다 허락해 줄게!"라는 뜻이에요. (모두 환영)
    allow_credentials=True, # 쿠키나 인증 정보도 주고받을 수 있게 허락해요.
    allow_methods=["*"], # 정보 읽기, 쓰기, 수정하기 등 모든 종류의 요청 방식을 다 허락해요.
    allow_headers=["*"], # 어떤 형태의 부가 정보(헤더)가 오든 다 받아줘요.
)

# ==========================================
# 4. 데이터 주문서 양식 만들기 (Pydantic)
# ==========================================
# 손님(웹사이트)이 주방(서버)에 질문을 보낼 때 지켜야 하는 양식이에요.
class QueryRequest(BaseModel):
    query: str                       # 손님의 실제 질문 내용 (문자열 형식이어야 함)
    chat_history: list               # 예전 대화 기록 (리스트 형식이어야 함)
    session_id: Optional[str] = None # 대화방 번호표 (선택사항이라 없어도 괜찮아요)
    config: dict                     # 여러 가지 설정값들 (사전 형식이어야 함)

# 주방(서버)이 요리를 끝내고 손님(웹사이트)에게 결과를 돌려줄 때 사용하는 양식이에요.
class QueryResponse(BaseModel):
    answer: str    # 인공지능이 만든 최종 답변 내용
    elapsed: float # 대답을 만드는 데 걸린 시간 (초)
    docs: list     # 답변을 만들 때 참고한 제안서 원문 조각들

# ==========================================
# 5. 서버가 켜질 때 딱 한 번! 기본 세팅하기
# ==========================================
# 💡 서버가 질문을 받을 때마다 모델을 부르면 너무 느려지니까, 가게 문을 열기 전에 미리 준비해 두는 과정이에요.

project_root = get_project_root_dir()                # 프로젝트 최상단 폴더를 찾아요.
config = load_config(project_root)                   # 설정 파일을 읽어와요.
dotenv_path = os.path.join(project_root, ".env")     # 비밀번호 파일(.env)의 위치를 찾아서...
load_dotenv(dotenv_path=dotenv_path)                 # 짠! 읽어옵니다.

# 글자를 숫자로 바꾸는 '임베딩 모델'을 준비해요.
embed_model_name = config["embedding"]["embed_model"]
embeddings = generate_embedding(embed_model_name)

# 답변을 글짓기 해줄 '생성 모델(인공지능 뇌)'을 준비해요.
model_type = config["generator"]["model_type"]
model_name = config["generator"]["model_name"]
use_quantization = config["generator"]["use_quantization"]
model_info = get_generation_model(model_type, model_name, use_quantization) # 모델을 메모리에 딱 올려둡니다!

# ==========================================
# 6. 실제 질문을 받는 창구(API 엔드포인트) 만들기
# ==========================================
# @app.post("/chat")은 손님이 인터넷 주소창 끝에 "/chat"을 붙여서 데이터를 보내면 아래 함수가 실행된다는 뜻이에요.
# response_model=QueryResponse는 "내가 돌려줄 답장은 아까 만든 QueryResponse 양식에 맞춰서 줄게"라는 약속이에요.
@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest): # 손님이 보낸 데이터 꾸러미를 'request'라는 이름으로 받아요.
    try:
        # 1️⃣ 원본 설정값이 바뀌지 않게 안전하게 복사본을 하나 만들어요.
        local_config = deepcopy(request.config)
        # 손님이 보낸 과거 대화 기록을 복사본 설정에 쏙 넣어줘요.
        local_config["chat_history"] = request.chat_history

        # 2️⃣ 예전 대화 기록이 있는지 확인해요.
        if request.chat_history:
            # 대화가 있었다면, 예전 대화 내용을 짧게 요약해 와요.
            chat_summary = load_chat_history(local_config, model_info)
            # 검색할 질문을 "이전 대화 요약 + 이번 질문"으로 합쳐서 더 똑똑하게 문서를 찾게 해줘요.
            local_config["retriever"]["query"] = f"이전 질문 요약: {chat_summary}\n질문: {request.query}"
        else:
            # 예전 대화가 없었다면(첫 질문이라면) 그냥 손님의 질문 그대로 사용해요.
            local_config["retriever"]["query"] = request.query

        # 3️⃣ 대망의 핵심! 파이프라인(공장)을 돌려서 답변을 받아내요.
        # 복사한 설정, 임베딩, 대화기록, 뇌 모델 등을 다 집어넣고 돌립니다.
        docs, answer, elapsed = rag_pipeline(
            local_config, 
            embeddings, 
            request.chat_history, 
            model_info, 
            is_save=True,               # DB에 저장할까요? 네!
            session_id=request.session_id # 대화방 번호표도 줍니다.
        )

        # 4️⃣ 찾은 참고 문서(docs)들을 보기 좋게 딕셔너리 형태로 예쁘게 포장해요.
        docs_result = [
            {
                "metadata": doc.metadata,    # 문서의 제목, 페이지 번호 같은 부가 정보
                "content": doc.page_content  # 실제 제안서 안의 글자 내용
            } for doc in docs
        ]

        # 5️⃣ 마지막으로 아까 약속했던 QueryResponse 양식에 맞춰서 손님(웹사이트)에게 배달을 보냅니다!
        return QueryResponse(answer=answer, elapsed=elapsed, docs=docs_result)

    except Exception as e:
        # FastAPI에 적절한 HTTP 에러 응답
        # logger.exception("Chat 엔드포인트 실패")
        print("/chat 엔드포인트 실패")
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 실패: {e}",
        )