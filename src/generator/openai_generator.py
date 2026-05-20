# 소스코드 분석 (2026.05.20 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 이 파일은 앞서 봤던 hf_generator.py의 OpenAI 버전이에요. 같은 역할을 하지만, 모델을 직접 GPU에 올리는 게 아니라 OpenAI 서버에 API 요청을 보내서 답변을 받아오는 구조라서 훨씬 간단합니다.

# 📦 1) Import 부분
import os  # 운영체제 관련 모듈 - 여기선 환경변수(API 키)를 읽기 위해 사용
from typing import Dict  # 타입 힌트용 - 함수 인자/반환값이 '딕셔너리'임을 명시
from langsmith import trace  # LangSmith 트레이싱 - OpenAI 호출을 로깅/모니터링하는 도구
from openai import OpenAI  # OpenAI 공식 Python SDK - GPT 모델에 API 요청을 보내는 클라이언트 클래스

# 💡 hf_generator.py와 비교하면 torch, transformers, BitsAndBytesConfig 같은 무거운 라이브러리가 전부 빠졌어요. 모델을 로컬에 띄울 필요가 없기 때문입니다. OpenAI 서버가 알아서 처리해주니까요.

# 🎯 핵심 흐름 한 줄 요약

# "환경변수에서 API 키 읽고 → OpenAI 클라이언트 만들고 → user 메시지로 prompt 보내고 → 답변 받아서 후처리하고 → 반환"

# hf_generator.py의 복잡한 텐서 처리, 디바이스 이동, 디코딩 작업이 전부 API 호출 한 줄(client.chat.completions.create(...))로 대체된 게 가장 큰 차이예요.
# 추가로 궁금한 부분(예: messages에 system 역할을 추가하면 어떻게 되는지, temperature 값에 따라 답변이 어떻게 달라지는지 등) 있으면 더 자세히 설명해드릴게요!


# 🔧 2) load_openai_model 함수 - 클라이언트 초기화
# 🔑 hf_generator의 load_hf_model은 수 GB짜리 모델 가중치를 GPU에 로딩하느라 몇 분이 걸릴 수 있지만, 이 함수는 그냥 API 키 확인 + 클라이언트 객체 생성만 하므로 1초도 안 걸립니다.
def load_openai_model(config: Dict) -> Dict:
    """
    OpenAI 모델명을 받아 초기화합니다.
    """
    # 환경변수에서 OpenAI API 키를 읽어옴 (보안을 위해 코드에 직접 적지 않음)
    # 보통 .env 파일이나 터미널에서 export OPENAI_API_KEY=sk-... 로 설정
    api_key = os.getenv("OPENAI_API_KEY")
    
    # API 키가 없거나 빈 문자열이면 에러 발생시켜 즉시 중단
    # → 나중에 호출 시점에 에러나는 것보다 로딩 단계에서 잡는 게 디버깅에 유리
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY가 설정되어 있지 않습니다.")

    # OpenAI 클라이언트 객체 생성 - 이후 이 객체로 API 요청을 보냄
    # 한 번 만들어두면 여러 번 재사용 가능 (연결 풀링 등 내부 최적화 효과)
    client = OpenAI(api_key=api_key)

    # 모델 정보를 딕셔너리로 묶어서 반환
    # → 이후 generate 함수가 이 dict만 받으면 모든 정보를 알 수 있음
    return {
        "type": "openai",                              # 어떤 종류의 백엔드인지 표시 (HF와 구분)
        "model": config["generator"]["model_name"],    # 사용할 모델명 (예: "gpt-4.1-nano")
        "client": client                               # 위에서 만든 OpenAI 클라이언트
    }
    
# 💬 3) generate_answer_openai 함수 - 답변 생성
def generate_answer_openai(prompt: str, model_info: Dict, generation_config: Dict) -> str:
    """
    OpenAI ChatCompletion API를 사용하여 프롬프트에 응답을 생성합니다.
    """
    # LangSmith 트레이싱 시작 - 이 블록 안의 모든 호출과 결과를 자동 기록
    # 디버깅, A/B 테스트, 성능 모니터링에 활용
    with trace(name="generate_answer_openai", inputs={"prompt": prompt}) as run:
        # load_openai_model이 만든 딕셔너리에서 클라이언트와 모델명을 꺼냄
        client = model_info["client"]
        model = model_info["model"]

        # OpenAI ChatCompletion API 호출 - 핵심 부분
        response = client.chat.completions.create(
            model=model,  # 사용할 모델 ID (예: "gpt-4.1-nano")
            
            # messages: 대화 형식의 입력 (역할/내용 쌍의 리스트)
            # "role": "user"는 사용자 발화임을 의미 (다른 옵션: "system", "assistant")
            # 여기서는 단일 턴이므로 user 메시지 하나만 전달
            messages=[{"role": "user", "content": prompt}],
            
            # 응답으로 생성할 최대 토큰 수 (없으면 기본 512)
            # HF의 max_new_tokens와 같은 역할
            max_tokens=generation_config.get("max_length", 512),
            
            # 0.0 = 가장 결정론적 (같은 입력 → 항상 같은 출력)
            # 값이 높을수록 창의적/무작위 (보통 0.0 ~ 2.0)
            # RAG에서는 일관성 위해 보통 0.0 사용
            temperature=0.0,
        )

        # response 구조:
        # response.choices = [Choice 객체 리스트] (n=1이면 1개만)
        # choices[0].message.content = 실제 답변 텍스트
        # .strip()으로 앞뒤 공백/줄바꿈 제거
        answer = response.choices[0].message.content.strip()
        
        # ===== 후처리 필터 (hf_generator와 동일한 로직) =====
        # 한국어 LLM이 자주 내뱉는 무의미한 어구를 제거
        bad_tokens = ["하십시오", "하실 수", "알고 싶어요", "하는데 필요한", "것을", "한다", "하십시오.", "하시기 바랍니다"]
        for token in bad_tokens:
            answer = answer.replace(token, "")  # 단순 문자열 치환으로 제거

        # 답변이 너무 짧거나(10자 미만), 단어 수가 너무 적은 경우(공백 3개 미만)
        # → 의미 있는 응답이 아닐 가능성 높으므로 폴백 메시지로 교체
        if len(answer) < 10 or answer.count(" ") < 3:
            answer = "해당 문서에서 예약 방법에 대한 명확한 정보를 찾을 수 없습니다."

        # LangSmith에 최종 출력 기록 (모니터링/디버깅용)
        run.add_outputs({"output": answer})
        
        # 정제된 최종 답변을 호출자에게 반환
        return answer