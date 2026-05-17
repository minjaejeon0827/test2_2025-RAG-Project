# 소스코드 분석 (2026.05.18 minjae)
# 참고: https://gemini.google.com/app/7192578f5d43d984?hl=ko

# 안녕하세요! 첨부해주신 RFPilot 프로젝트의 README 파일과 chat_history.py 코드를 모두 꼼꼼하게 읽고 분석했습니다.

# 매일 쏟아지는 방대한 B2G 제안요청서(RFP)를 다루기 위해 팀원 분들과 함께 정말 탄탄한 RAG(검색 증강 생성) 파이프라인을 구축하셨네요. 이번에 보여주신 chat_history.py 파일은 챗봇이 사용자와 이전에 나누었던 대화의 맥락을 잊어버리지 않도록 '기억력을 관리해 주는 핵심 매니저' 역할을 하는 코드입니다.

# 대화가 길어지면 컴퓨터가 모든 내용을 다 기억하기에는 너무 무겁고 느려지기 때문에, 이 코드는 과거 대화 내용을 아주 짧고 굵게 '요약'해서 똑똑하게 기억하는 방식을 사용하고 있습니다.

# 이번에도 중학생 친구에게 설명하듯 쉽고 친절하게 한 줄 한 줄 주석을 달아드릴게요. 천천히 읽어보세요!

# 💡 핵심 요약!
# 이 파일의 목적은 챗봇이 "아까 말했던 조건으로 다시 찾아줘" 같은 질문을 받았을 때 당황하지 않게 만드는 것입니다. 대화가 누적될수록 데이터가 기하급수적으로 커지는 것을 막기 위해, 대화 원문을 그대로 넘기지 않고 LLM(대규모 언어 모델)을 이용해 대화의 핵심만 쏙쏙 뽑아 요약(Summarize)한 뒤 넘겨준다는 점이 이 시스템의 가장 똑똑한 부분입니다.

# 프로젝트 전반적으로 시스템 최적화와 메모리 관리에 많은 신경을 쓰신 것이 돋보입니다. 코드를 살펴보시면서 더 궁금한 부분이 생기면 언제든 질문 남겨주세요!

# ==========================================
# 1. 외부 도구(다른 폴더의 파일들) 가져오기
# ==========================================
# 답변을 만들어내는 인공지능 '뇌' 역할을 하는 함수들을 가져옵니다.
# 하나는 허깅페이스(예: 팀에서 쓰는 Gukbap 모델), 하나는 오픈AI(예: GPT 모델)용이에요.
from src.generator.hf_generator import generate_answer_hf
from src.generator.openai_generator import generate_answer_openai

# ==========================================
# [함수 1] 과거 대화 내용을 짧게 요약해 주는 함수
# ==========================================
def summarize_chat_history(config, model_info=None):
    """
    질의응답 내역 요약을 생성하는 함수입니다.
    지금까지 챗봇과 나눈 대화가 너무 길면 무거워지니까, 
    인공지능에게 "이전 대화 내용 좀 3줄 요약해 줘!"라고 시키는 과정이에요.
    """
    
    # 1. 인공지능 모델(뇌)이 준비되어 있는지 확인해요. 없으면 에러를 냅니다!
    if model_info is None:
        raise ValueError("❌ (generator.chat_history.summarize_chat_history) model_info가 없습니다.")

    # 2. 기억 수첩(config["chat_history"])에 적힌 대화들을 쭉 꺼내옵니다.
    # 사용자가 말한 건 "질문: ~", AI가 말한 건 "답변: ~" 형식으로 예쁘게 묶어서 하나의 긴 글로 만들어요.
    history_text = "\n".join(
        [f"{'질문' if turn['role'] == 'user' else '답변'}: {turn['content']}" for turn in config["chat_history"]]
    )
    
    # 3. 인공지능에게 내릴 명령서(프롬프트)를 작성해요.
    # "이 대화의 핵심 내용을 간결하게 요약해 줘"라는 말 뒤에 방금 묶어둔 대화 내용을 찰싹 붙여줍니다.
    prompt = f"다음은 사용자와 AI의 대화 내용입니다. 이 대화의 핵심 내용을 간결하게 요약해 주세요.\n\n{history_text}"

    # 4. 설정 파일에서 어떤 인공지능을 쓰기로 했는지 확인하고, 요약을 부탁해요.
    if config["generator"]["model_type"] == "huggingface":
        # 허깅페이스 모델을 쓴다면 이쪽 공장으로 명령서를 보냅니다.
        return generate_answer_hf(prompt, model_info, config["generator"])
    elif config["generator"]["model_type"] == "openai":
        # 오픈AI 모델을 쓴다면 이쪽 공장으로 명령서를 보냅니다.
        return generate_answer_openai(prompt, model_info, config["generator"])
    

# ==========================================
# [함수 2] 요약된 대화 기록을 불러와서 전달해 주는 함수
# ==========================================
def load_chat_history(config, model_info=None):
    """
    질의응답 내역 요약을 로드하는 함수입니다.
    질문이 들어왔을 때 가장 먼저 실행되어서, 예전 대화 기록이 있는지 없는지 검사하는 '문지기' 역할이에요.
    """
    
    # 1. 기억 수첩(config["chat_history"])에 예전에 나눈 대화가 단 하나라도 적혀 있다면?
    if config["chat_history"]:
        
        # (이 줄은 대화 기록을 문자열로 만드는 건데, 바로 아래 줄에서 요약본으로 덮어씌워지기 때문에 사실상 임시 작업이에요)
        chat_history_str = "\n".join([f"질문: {turn['content']}" for turn in config["chat_history"]])

        # 2. 방금 위에서 만든 '요약 전문가(summarize_chat_history)' 함수를 불러와서 요약본을 받아옵니다.
        chat_history_str = summarize_chat_history(config, model_info)
        
        # 3. 개발자가 잘 돌아가는지 볼 수 있게 터미널 화면에 요약된 내용을 띄워줘요.
        print(f"과거 대화 내역 요약: {chat_history_str}")
        
        # 4. 챗봇이 다음 대답을 할 때 참고할 수 있도록 이 요약본을 넘겨줍니다.
        return chat_history_str
        
    # 5. 만약 수첩이 텅텅 비어 있다면? (즉, 사용자가 지금 막 첫 질문을 한 상태라면)
    else:
        # 기억할 게 없으니 그냥 텅 빈 글자(빈 문자열)를 만들어서 넘겨줍니다.
        chat_history_str = ""
        return chat_history_str