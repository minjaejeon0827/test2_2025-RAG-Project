# 소스코드 분석 (2026.05.19 minjae)
# 참고: https://gemini.google.com/app/7192578f5d43d984?hl=ko

# 안녕하세요! 첨부해주신 RFPilot 프로젝트의 README 파일과 generator_main.py 코드를 꼼꼼하게 모두 읽고 분석했습니다.

# README를 보니 정말 멋진 프로젝트네요! 매일 쏟아지는 수십 페이지의 B2G 제안요청서(RFP)를 인공지능이 대신 읽고 요약해 준다니, 실무에 당장 투입해도 될 만큼 훌륭한 아이디어입니다. 타임라인을 보니 '제너레이터 통합 및 구현'은 정재의 님께서 주로 담당하셨던 것 같네요. 5명의 팀원분들이 각자의 역할을 맡아 성공적으로 파이프라인을 완성해 나가시는 모습이 대단합니다!

# 이전 파일들이 문서를 찾고 정리하는 '도서관 사서' 역할이었다면, 이번에 보여주신 generator_main.py는 찾아온 자료들을 바탕으로 사용자에게 예쁘게 말을 건네는 '최종 대변인(또는 작가)' 역할을 하는 아주 중요한 코드입니다.

# 이번에도 중학생 친구에게 설명하듯 어려운 전문 용어는 쏙 빼고, 아주 쉽고 친절하게 한 줄 한 줄 주석을 달아드릴게요. 편안하게 읽어보세요! 😊

# 💡 핵심 요약!
# 이 코드는 RAG(검색 증강 생성) 시스템의 가장 마지막 단계를 책임지는 아주 깔끔한 교통정리 코드입니다.

# 사용자의 질문, 찾아온 문서, 그리고 과거의 대화 기록까지 싹 모아서 "이 재료들을 바탕으로 대답해 줘!"라는 하나의 거대한 편지(프롬프트)를 만든 다음, 우리가 선택한 인공지능(OpenAI 또는 HuggingFace)에게 보내서 진짜 사람 같은 답변을 받아내는 훌륭한 역할을 하고 있습니다.

# 코드의 흐름이 물 흐르듯 아주 논리적이고 직관적이네요. 팀원분들과 함께 정말 고생 많으셨습니다. 더 궁금한 코드가 있다면 언제든 편하게 알려주세요!

# ==========================================
# 1. 외부 도구(라이브러리) 및 내부 도구 가져오기
# ==========================================
from typing import List # "이건 여러 개가 들어있는 상자(리스트)야!"라고 파이썬에게 힌트를 주는 도구예요.
from langchain.schema import Document # 글자들을 깔끔하게 담아두는 '문서 상자' 도구예요.

# 우리 팀이 만든 다른 폴더의 파일들을 가져와요.
from src.generator.hf_generator import generate_answer_hf # 허깅페이스(무료 AI 창고) 모델로 대답을 만드는 공장이에요.
from src.generator.openai_generator import generate_answer_openai # 오픈AI(GPT 등) 모델로 대답을 만드는 공장이에요.
from src.generator.make_prompt import build_prompt # 인공지능에게 "이렇게 대답해!"라고 지시할 '명령서'를 예쁘게 써주는 도구예요.

'''
    TODO: (앞으로 해야 할 일 메모장)
    - 변경된 모듈들에 맞춰서 generator_main 함수 수정 ✅ (완료했음!)
    - load_chat_history 함수 위치 변경에 의한, import 경로 수정 ✅ (완료했음!)
'''

# ==========================================
# 2. 🌟 메인 핵심 함수: 최종 답변 글짓기
# ==========================================
def generator_main(
    retrieved_docs:List[Document], # 사서가 도서관에서 열심히 찾아온 '참고 자료(문서 조각)'들이에요.
    config:dict, # 프로젝트의 모든 설정이 적혀있는 '설정 수첩'이에요.
    model_info:dict = None, # 대답을 만들어낼 똑똑한 '인공지능 뇌'에 대한 정보예요.
    chat_history = None # 예전에 사용자와 나눴던 '과거 대화 기록'이에요.
) -> str: # 최종적으로 만들어진 깔끔한 '문장(텍스트)'을 결과물로 내보낼 거예요.
    """
    검색된 문서 리스트를 기반으로 답변을 생성하는 메인 실행 함수.
    (도서관에서 찾은 자료를 인공지능에게 주면서 대답을 만들어오라고 시키는 감독관 역할입니다!)
    """
    
    # 1. 인공지능 뇌가 제대로 준비되었는지 확인해요.
    if model_info is None:
        # 뇌가 없으면 일을 할 수 없으니 에러를 내고 멈춥니다!
        raise ValueError("❌ (generator.generator_main.generator_main) model_info가 없습니다.")

    # 2. 사용자가 맨 처음 물어봤던 '질문'을 설정 수첩에서 꺼내와요.
    query = config["retriever"]["query"]

    # 3. 인공지능에게 건네줄 '완벽한 명령서(프롬프트)'를 작성해요.
    # 아까 가져온 build_prompt라는 도구를 써서 빈칸을 채워 넣습니다.
    prompt = build_prompt(
        question=query, # 사용자의 질문
        retrieved_docs=retrieved_docs, # 사서가 찾아온 참고 자료들
        include_source=config.get("include_source", True), # 답변 밑에 출처를 남길지 말지 결정해요.
        prompt_template=config.get("prompt_template"), # 우리 팀이 미리 짜둔 명령서 기본 양식을 써요.
        chat_history=chat_history, # 챗봇이 맥락을 까먹지 않게 과거 대화 기록도 같이 줍니다.
    )

    # 4. 설정 수첩을 보고 우리가 어떤 종류의 인공지능을 쓰기로 했는지 확인해요.
    model_type = config["generator"]["model_type"]
            
    # 5. 확인한 종류에 맞춰서 딱 맞는 공장으로 명령서를 보냅니다.
    if model_type == "huggingface":
        # 허깅페이스 모델이면 여기로 보내서 답변(answer)을 받아와요!
        answer = generate_answer_hf(prompt, model_info, config["generator"])
    elif model_type == "openai":
        # 오픈AI 모델이면 저기로 보내서 답변(answer)을 받아와요!
        answer = generate_answer_openai(prompt, model_info, config["generator"])

    # 6. 개발자가 잘 돌아가는지 볼 수 있게 까만 터미널 창에 중간 점검 내용을 띄워줘요.
    print(f"✅ 과거 맥락 :{chat_history}")
    print(f"✅ 답변 :{answer}")

    print(answer) # 한 번 더 깔끔하게 대답만 출력해주고,
    print("✅ 답변 생성 완료") # 다 끝났다고 도장을 꽝! 찍어줍니다.
    
    # 7. 완성된 최종 대답을 화면(Streamlit) 쪽으로 슝~ 하고 던져줍니다!
    return answer