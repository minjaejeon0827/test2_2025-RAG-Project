# 소스코드 분석 (2026.06.03 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 🍽️ 들어가기 전에
# 이 main.py는 RFP 챗봇이라는 퓨전 레스토랑의 총괄 매니저예요. 주방의 각 부서(loader=재료 손질, embedding=재료 보관, retriever=재료 찾기, generator=요리)에 차례대로 일을 시켜서 손님 질문에 답을 만들어내는 사령탑이에요. 자, 이제 매니저의 일을 한 줄씩 따라가 봅시다!

# 📥 1부: 외부 도구(라이브러리) 가져오기
import os     # 컴퓨터 안의 폴더/파일 경로를 다룰 때 쓰는 기본 도구예요.
import time   # 시간을 재는 도구. "이 작업 몇 초 걸렸지?" 측정할 때 써요.

from typing import Dict           # "이 함수는 딕셔너리를 반환해" 라고 파이썬에게 알려주는 힌트 도구예요.
from langsmith import trace       # AI가 일하는 모든 발자국을 기록해주는 GPS 같은 도구예요.
from dotenv import load_dotenv    # 비밀번호(.env 파일)을 안전하게 읽어오는 도구예요.

# 📥 2부: 우리 프로젝트 내부 부서들 호출 준비
# 각 부서의 "메인 진입점" 함수들을 호출할 수 있게 가져와요. 매니저는 이 함수들만 알면 돼요.

from src.loader.loader_main import loader_main                # 🥕 재료 손질 부서장 (PDF/HWP → 텍스트 조각)
from src.embedding.embedding_main import embedding_main       # 🧊 재료 보관 부서장 (텍스트 → 벡터 DB)
from src.embedding.vector_db import generate_embedding        # 🔤→🔢 글자를 숫자로 바꾸는 번역기 만들기
from src.retrieval.retrieval_main import retrieval_main       # 🔍 재료 찾기 부서장 (질문 → 관련 문서)
from src.utils.config import load_config                      # 📜 설정 파일(config.yaml) 읽어오기
from src.utils.path import get_project_root_dir               # 📂 프로젝트 최상단 폴더 위치 찾기
from src.generator.generator_main import generator_main       # 🍳 요리 부서장 (문서 → 답변)
from src.generator.chat_history import load_chat_history      # 💬 이전 대화 기록 불러오기
from src.generator.hf_generator import load_hf_model          # 🤗 HuggingFace 모델 로더
from src.generator.openai_generator import load_openai_model  # 🤖 OpenAI 모델 로더

# 🏭 3부: get_generation_model 함수 — LLM 모델 공장
def get_generation_model(model_type: str, model_name: str, use_quantization: bool = False) -> Dict:
    # ↑ 함수 정의: "어떤 LLM 모델을 쓸지 알려주면, 그 모델을 준비해서 돌려줄게"
    # model_type: "huggingface" 또는 "openai" 중 하나
    # model_name: 정확한 모델 이름 (예: "gpt-4.1-nano")
    # use_quantization: 모델을 가볍게 만드는 기술 사용 여부 (기본값: 안 씀)
    # -> Dict: 모델 정보를 딕셔너리 형태로 돌려줄 거예요
    
    """
    지정된 모델 타입 및 이름에 따라 생성 모델을 로드합니다.
    (위는 docstring — 다른 개발자에게 "이 함수 어떻게 쓰는지" 알려주는 설명서)
    """
    
    # 받은 인자 3개를 깔끔하게 딕셔너리로 포장해요. 
    # 왜냐하면 아래에서 호출할 load_hf_model 같은 함수들이 이 형식을 기대하거든요.
    config = {
        'generator': {                              # "생성기" 섹션을 만들고
            'model_type': model_type,                # 모델 종류 저장
            'model_name': model_name,                # 모델 이름 저장
            'use_quantization': use_quantization     # 양자화 여부 저장
        }
    }

    # 분기 처리! 어떤 모델이냐에 따라 다른 로더를 호출해요.
    if model_type == 'huggingface':
        return load_hf_model(config)        # 🤗 HuggingFace면 이 함수 호출 후 결과 반환
    elif model_type == 'openai':
        return load_openai_model(config)    # 🤖 OpenAI면 이 함수 호출 후 결과 반환
    else:
        # 둘 다 아니면? 우리가 모르는 모델이니까 에러 발생시키고 멈춰요!
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
    
# 🍳 4부: rag_pipeline 함수 — 요리의 전체 순서 (이 파일의 핵심!)
def rag_pipeline(config, embeddings, chat_history, model_info=None, is_save=False, session_id=None):
    # ↑ RAG(검색 증강 생성) 파이프라인 전체를 한 번에 돌리는 함수예요.
    # config: 전체 설정 꾸러미
    # embeddings: 글자→숫자 번역기
    # chat_history: 이전 대화 기록 (이전에 뭘 물어봤는지 기억)
    # model_info: 미리 준비된 LLM 모델 정보
    # is_save: 벡터 DB를 새로 저장할지? (True=새로 만들기, False=기존 거 불러오기)
    # session_id: 지금 대화방의 고유 번호표 (예: "a3f9b2c1")
    
    try:    # 🛡️ "이 안의 코드들이 에러나면 안전하게 잡을게" (try-except 안전장치)
    
        # 🛰️ 전체 RAG 파이프라인 추적 시작! GPS를 켠 것처럼 LangSmith에 기록돼요.
        with trace(name="rag_pipeline") as run:

            # ============================================================
            # 🥕 [단계 1] 재료 손질하기 (문서 로딩 + 청크 분할)
            # ============================================================
            with trace(name="loader_main"):    # 이 단계도 따로 추적
                # loader_main 호출: PDF/HWP 파일을 텍스트로 추출하고 잘게 자른 조각(chunks)으로 만들어요.
                # 결과 예시: ["사업명은 ...", "예산은 130,000,000원...", ...]
                chunks = loader_main(config, embeddings, chat_history)
                
            # ============================================================
            # 🧊 [단계 2] 재료를 냉장고에 라벨 붙여 보관 (임베딩 + 벡터 DB)
            # ============================================================
            with trace(name="embedding_main"):
                # 글자 조각들을 숫자(벡터)로 바꾸고, 검색 가능한 똑똑한 DB에 저장
                # is_save=True면 새로 생성, False면 기존 DB 로딩
                vector_store = embedding_main(
                    config, 
                    chunks,                       # 위에서 만든 텍스트 조각들
                    embeddings=embeddings,        # 번역기
                    is_save=is_save,              # 저장 모드 여부
                    session_id=session_id         # 대화방 번호표 (DB 이름에 붙임)
                )

            # ============================================================
            # 🔍 [단계 3] 손님 주문에 맞는 재료 찾기 (관련 문서 검색)
            # ============================================================
            with trace(name="retrieval_main"):
                # 사용자 질문과 가장 비슷한 문서들을 벡터 DB에서 골라옴
                # 결과 예시: 상위 5개 관련 문서 리스트
                docs = retrieval_main(
                    config, 
                    vector_store,                 # 위에서 만든/불러온 벡터 DB
                    chunks,                       # 원본 조각들 (앙상블 검색용)
                    embeddings=embeddings,        # 번역기
                    chat_history=chat_history     # 이전 대화 (맥락 반영)
                )

            # ============================================================
            # 🍳 [단계 4] 요리 시작! (답변 생성 + 시간 측정)
            # ============================================================
            with trace(name="generator_main"):
                start_time = time.time()    # ⏱️ 시작 시각 기록 (현재 시각을 초 단위로)
                
                # generator_main 호출: 검색된 문서를 LLM에게 주고 답변 받아오기
                answer = generator_main(
                    docs,                         # 검색된 문서들
                    config, 
                    model_info=model_info,        # 미리 준비된 LLM 모델
                    chat_history=chat_history     # 이전 대화 (자연스러운 흐름용)
                )
                
                end_time = time.time()                                  # ⏱️ 종료 시각 기록
                elapsed = round(end_time - start_time, 2)               # 걸린 시간 계산 (소수점 2자리, 예: 2.45초)

            # ============================================================
            # 📊 [단계 5] LangSmith에 결과 기록 (나중에 분석용)
            # ============================================================
            run.add_outputs({                                       # 이번 실행 결과를 한꺼번에 저장
                "query": config["retriever"]["query"],              # 사용자가 물어본 질문
                "model_type": config["generator"]["model_type"],    # 사용한 모델 종류
                "model_name": config["generator"]["model_name"],    # 사용한 모델 이름
                "max_length": config["generator"]["max_length"],    # 최대 답변 길이 설정값
                "num_chunks": len(chunks),                          # 잘린 조각 개수 (예: 1000개)
                "num_retrieved_docs": len(docs),                    # 검색된 관련 문서 개수 (예: 5개)
                "answer_length": len(answer),                       # 답변 길이 (글자 수)
                "final_answer": answer                              # 최종 답변 내용 자체
            })
            
            # ============================================================
            # 📦 [단계 6] 결과 3개를 한 번에 반환!
            # ============================================================
            return docs, answer, elapsed    # (검색된 문서들, 답변, 걸린 시간)
            
    except Exception as e:    # 위 try 블록에서 에러가 나면 여기로 와요
        print(f"❌ 로깅 에러: {e}")    # 에러 메시지 출력하고 끝
        # ⚠️ 여기 약점: return이 없어서 None 반환됨. 호출자가 언팩하면 에러 발생!
        raise
        
        
# 🚀 5부: if __name__ == "__main__": — 직접 실행 영역
if __name__ == "__main__":
    # ↑ "이 파일을 직접 'python main.py'로 실행했을 때만 아래 코드 돌려라" 라는 의미
    # 다른 파일에서 import만 하면 아래는 안 돌아요. (안전장치)
    
    # 📂 1. 프로젝트 루트 폴더 찾기 (예: D:/codeit/)
    project_root = get_project_root_dir()

    # 📜 2. config.yaml과 .env 파일 경로 만들기
    config_path = os.path.join(project_root, "config.yaml")     # ⚠️ 만든 후 안 씀! 불필요한 줄
    dotenv_path = os.path.join(project_root, ".env")             # .env 파일 경로 조립
    load_dotenv(dotenv_path=dotenv_path)                         # .env에서 API 키 등 환경변수 로딩!

    # ⚙️ 3. 설정 파일(config.yaml) 로딩 + 검증 → 딕셔너리로 받기
    config = load_config(project_root)

    # 📦 4. 설정에서 필요한 값들을 변수로 꺼내요 (택배 박스에서 물건 꺼내듯)
    embed_model_name = config["embedding"]["embed_model"]       # 예: "openai"
    model_type = config["generator"]["model_type"]               # 예: "openai"
    model_name = config["generator"]["model_name"]               # 예: "gpt-4.1-nano"
    use_quantization = config["generator"]["use_quantization"]   # 예: False
    
    # 🏭 5. 무거운 작업들 (각 모델 로딩 — 수십 초~수 분 걸릴 수 있음!)
    embeddings = generate_embedding(embed_model_name)            # 🔤→🔢 글자→숫자 번역기 준비
    model_info = get_generation_model(model_type, model_name, use_quantization)  # 🤖 LLM 모델 로딩
    chat_history = load_chat_history(config, model_info)         # 💬 이전 대화 기록 불러오기
    
    # 🎬 6. 드디어 RAG 파이프라인 실행!
    # is_save=True니까 벡터 DB를 새로 만들어서 저장하면서 답변 생성
    rag_pipeline(config, embeddings, chat_history, model_info, is_save=True)