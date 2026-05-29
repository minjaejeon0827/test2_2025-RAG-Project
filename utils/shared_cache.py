# 소스코드 분석 (2026.05.29 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 이 파일은 짧지만 모든 모델 다운로드를 어디에 저장할지 미리 정해주는 설정 함수예요. chatbot.py 맨 위에서 set_cache_dirs()로 호출되어, AI 모델들이 사용하는 캐시 폴더 경로를 통일시켜 줍니다.

# 🔑 핵심 개념 3가지
# 1) os.environ이란?
# 운영체제의 환경변수를 다루는 딕셔너리예요. 환경변수란 OS 전체가 공유하는 설정값으로, 라이브러리들은 이 값을 보고 동작 방식을 결정합니다.
# os.environ['TRANSFORMERS_CACHE'] = '/some/path'
# → 이후 HuggingFace 라이브러리가 모델을 받을 때 이 경로를 보고
#    "아, 여기에 저장하라는 거구나" 하고 따름

# 2) os.path.expanduser('~/...')의 마법
# ~(틸드)는 셸에서 "현재 사용자의 홈 디렉토리"를 뜻하지만, 파이썬 함수에 그대로 넘기면 문자 그대로 처리됩니다. 그래서 expanduser로 실제 경로로 변환해줘야 해요.

# OS | ~ 변환 결과
# 리눅스 | /home/사용자명
# 맥 | /Users/사용자명
# 윈도우 | C:\Users\사용자명

# 예: ~/.cache/huggingface/transformers → /home/user/.cache/huggingface/transformers
# 3) os.path.abspath와의 차이

# expanduser('~/.cache'): 홈 디렉토리 기준 → 사용자별 공통 캐시
# abspath("2025-LLM-Project/.cache"): 현재 작업 디렉토리 기준 절대 경로

# 마지막 HF_HOME 줄만 abspath를 써서 프로젝트 폴더 안에 저장하려고 한 거예요.

# 🤖 각 캐시 경로가 어떤 모델을 저장하는지
# 지금까지 본 파일들과 연결해 보면:

# 환경변수 | 영향 받는 코드 | 저장되는 것
# TRANSFORMERS_CACHE | hf_generator.py의 AutoModelForCausalLM.from_pretrained(...) | Qwen2.5-7B 가중치
# SENTENCE_TRANSFORMERS_HOME | data_loader.py의 KoE5 임베딩, retrieval.py의 CrossEncoder | KoE5, ms-marco-MiniLM
# TORCH_HOME | PyTorch 사전학습 모델 | torch 체크포인트
# HF_HOME | HuggingFace Hub 전반 | 모델/데이터셋/토큰 등

# ⚠️ "문제시 삭제" 주석의 의미
# 마지막 줄에 주석으로 # 문제시 삭제가 붙은 이유는 설정 충돌 가능성 때문이에요. 정리하면:
# 위 5줄은 모두 ~/.cache/... (홈 디렉토리 기준)
# os.environ['TRANSFORMERS_CACHE'] = '~/.cache/huggingface/transformers'
# ...

# 마지막 줄만 프로젝트 폴더 기준
# os.environ["HF_HOME"] = os.path.abspath("2025-LLM-Project/.cache")

# HF_HOME은 HuggingFace의 상위 통합 설정이라, TRANSFORMERS_CACHE보다 우선순위가 높을 수도 있어요. 또 abspath("2025-LLM-Project/.cache")는 실행 위치가 잘못되면 엉뚱한 경로가 될 수 있고요(예: 다른 폴더에서 실행하면 /현재폴더/2025-LLM-Project/.cache가 됨).
# 그래서 개발자가 "만약 캐시 관련 문제가 생기면 이 줄부터 지워봐"라는 메모를 남긴 거예요.

# 🎯 왜 이런 함수가 필요할까?
# 문제 상황: 캐시 경로를 안 지정하면 어떻게 될까요?
# ✗ 사용자A의 홈 디렉토리에 7GB 모델 다운로드
# ✗ 사용자B는 또 다시 7GB 다운로드
# ✗ 사용자C는 노트북 용량 부족으로 실패
# ✗ 같은 모델이 여러 폴더에 중복 저장

# 해결: 캐시 경로를 통일 → 한 번만 받고 모두가 공유
# ✓ ~/.cache/sbert/ 에 KoE5 한 번만 저장
# ✓ 다른 프로젝트에서도 같은 모델 재사용 가능
# ✓ 디스크 절약, 다운로드 시간 절약

# ⚡ 호출 시점이 중요해요
# chatbot.py에서 이 함수가 import 직후 바로 호출되는 게 핵심입니다:
# from src.utils.shared_cache import set_cache_dirs
# ...
# set_cache_dirs()  # ← 여기! 모델 import 전에 실행
# load_dotenv()
# torch.classes.__path__ = []

# 왜 먼저 실행해야 하나? 만약 다른 라이브러리(torch, transformers 등)가 먼저 import되면, 그것들이 import 시점에 자기 기본 캐시 경로를 미리 결정해버려서 나중에 환경변수를 바꿔도 안 먹힐 수 있거든요. "환경변수 설정 → 라이브러리 사용" 순서를 지켜야 합니다.

# 💡 한 줄 요약

# "AI 모델들이 다운로드받을 캐시 폴더 위치를 통일시켜, 디스크 낭비를 막고 여러 라이브러리가 같은 경로를 보게 만드는 사전 설정 함수"

# 추가로 궁금한 부분(예: 환경변수가 어떻게 자식 프로세스로 상속되는지, HF_HOME과 TRANSFORMERS_CACHE 중 어느 게 우선되는지, Docker 환경에서의 캐시 설정 등) 있으면 더 자세히 설명해드릴게요!

# 📦 Import
import os  # 환경변수 설정과 경로 조작에 필요

# 🗂️ set_cache_dirs 함수 한 줄씩 분석
def set_cache_dirs():
    # ===== TRANSFORMERS_CACHE =====
    # HuggingFace Transformers 라이브러리(AutoModelForCausalLM 등)가
    # 모델 가중치를 다운로드해서 저장하는 폴더
    # os.path.expanduser('~'): '~' 를 실제 홈 디렉토리 경로로 변환
    #   리눅스/맥: /home/사용자명, 윈도우: C:\Users\사용자명
    os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
    
    # ===== SENTENCE_TRANSFORMERS_HOME =====
    # sentence-transformers 라이브러리(KoE5 임베딩 모델 등)가
    # 모델 파일을 저장하는 폴더
    # → 앞서 본 retrieval.py의 CrossEncoder, data_loader.py의 embeddings도 여기 사용
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/sbert')
    
    # ===== XDG_CACHE_HOME =====
    # 리눅스 XDG 표준의 캐시 폴더 위치 (많은 라이브러리가 따르는 관례)
    # 일부 라이브러리가 이 환경변수를 보고 캐시 위치를 결정함
    os.environ['XDG_CACHE_HOME'] = os.path.expanduser('~/.cache/xdg')
    
    # ===== LANGCHAIN_CACHE =====
    # LangChain 라이브러리가 사용하는 캐시 경로
    # (LLM 호출 결과 캐싱 등에 활용)
    os.environ['LANGCHAIN_CACHE'] = os.path.expanduser('~/.cache/langchain')
    
    # ===== TORCH_HOME =====
    # PyTorch 가 사전학습 모델/체크포인트를 받아오는 폴더
    # (torchvision 등의 pretrained=True 옵션 사용 시 여기에 저장됨)
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

    # ===== HF_HOME (위 설정과 충돌 위험!) =====
    # 주석에 "문제시 삭제"라고 적혀 있는 줄
    # HF_HOME: HuggingFace Hub 전반(모델/데이터셋/토큰)의 통합 캐시 경로
    # → 위에서 TRANSFORMERS_CACHE는 홈 디렉토리(~/.cache/...)로 설정했는데,
    #   여기서는 프로젝트 폴더 안(2025-LLM-Project/.cache)으로 강제 지정
    # → 두 설정이 충돌해서 동작이 꼬일 수 있어 "문제시 삭제" 주석을 붙여둔 것
    os.environ["HF_HOME"] = os.path.abspath("2025-LLM-Project/.cache")