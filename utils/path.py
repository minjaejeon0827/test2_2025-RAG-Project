# 소스코드 분석 (2026.05.28 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 이 파일은 짧지만 모든 모듈이 의지하는 핵심 유틸리티예요. 앞서 본 chatbot.py, config.py 등에서 get_project_root_dir()를 호출해 프로젝트 루트 경로를 알아냈죠. 이 함수가 정확히 어떻게 그걸 찾아내는지 한 줄씩 설명드릴게요.

# 🎯 핵심 함수 4개 정리
# 이 코드의 모든 마법은 os 모듈의 4가지 함수에서 나와요. 예시 경로 /home/user/2025-LLM-Project/src/utils로 비교하면:
# 함수 | 역할 | 결과
# os.getcwd() 지금 실행 중인 폴더 /home/user/2025-LLM-Project/src/utils
# os.path.abspath(...) 절대 경로로 변환 위와 동일 (이미 절대경로)
# os.path.basename(...) 경로의 마지막 폴더 이름 utils
# os.path.dirname(...) 경로의 부모 디렉토리 /home/user/2025-LLM-Project/src

# 📈 실제 동작 예시 (단계별 추적)
# 만약 /home/user/2025-LLM-Project/src/utils/path.py에서 함수가 호출됐다면:
# 시작: cwd = /home/user/2025-LLM-Project/src/utils

# [루프 1회차]
#   basename(cwd) = "utils"
#   "utils" == "2025-LLM-Project"? ❌
#   부모로 이동: cwd = /home/user/2025-LLM-Project/src

# [루프 2회차]
#   basename(cwd) = "src"
#   "src" == "2025-LLM-Project"? ❌
#   부모로 이동: cwd = /home/user/2025-LLM-Project

# [루프 3회차]
#   basename(cwd) = "2025-LLM-Project"
#   "2025-LLM-Project" == "2025-LLM-Project"? ✅
#   → return "/home/user/2025-LLM-Project"   🎉

# 🛑 무한루프 방지 장치: while cwd != os.path.dirname(cwd)
# 이 조건이 정말 중요해요. 만약 프로젝트 폴더를 못 찾고 계속 올라가면 어떻게 될까요?
# 파일시스템 최상단의 특이점:
# OS | 루트 경로 | 부모 경로 (dirname) | 
# 리눅스/맥 | / | / (자기 자신!)
# 윈도우 | C:\  | C:\ (자기 자신!)
# 최상단에서는 부모와 자기 자신이 같아져요. 
# 그래서 cwd != os.path.dirname(cwd)가 False가 되면서 루프가 끝납니다. 
# 이 안전장치가 없으면 무한히 dirname을 호출하다가 무한루프에 빠질 수 있어요.

# ⚠️ 발생할 수 있는 시나리오
# 케이스 1: 정상 실행

# 프로젝트 안 어디서 실행하든 위로 올라가다 2025-LLM-Project 폴더를 찾으면 그 경로 반환 ✅

# 케이스 2: 프로젝트 밖에서 실행

# 예: /tmp/에서 실행하면 위로 계속 올라가도 2025-LLM-Project를 못 찾음
# → 루프가 /까지 가서 종료 → RuntimeError 발생 ❌

# 케이스 3: 폴더 이름이 다른 경우

# 누군가 프로젝트를 RFPilot/ 같은 다른 이름으로 받았다면 못 찾음
# 해결법: get_project_root_dir("RFPilot") 처럼 인자 전달

# 🎯 왜 이 함수가 필요한가?
# 파이썬 프로젝트에서 상대 경로의 함정을 피하기 위해서예요. 예를 들어:
# ❌ 나쁜 방법: 상대 경로 (실행 위치에 따라 깨짐)
# config = load_config("config.yaml")

# ✅ 좋은 방법: 절대 경로 (어디서 실행해도 동일)
# root = get_project_root_dir()
# config = load_config(os.path.join(root, "config.yaml"))

# chatbot.py를 python chatbot.py로 실행하든, python src/streamlit/chatbot.py로 실행하든, 또는 Streamlit으로 띄우든 — 항상 같은 절대 경로를 얻을 수 있어요. 이게 가능한 이유가 바로 이 작은 함수 덕분입니다.

# 🔄 다른 파일들과의 연결
# 지금까지 본 파일들에서 이 함수가 어떻게 쓰였는지 다시 보면:
# 즉, 프로젝트의 모든 경로 계산의 출발점이 되는 함수예요. 작지만 없으면 안 되는, 진짜 "기초공사" 같은 코드죠.
# chatbot.py (Streamlit 프론트)
# project_root = get_project_root_dir()        # ← 여기서 호출
# config = load_config(project_root)            # → 그 경로로 config.yaml 로딩

# # config.py
# config_path = os.path.join(project_root, "config.yaml")  # 절대 경로 조합

# 💡 한 줄 요약

# "지금 어디서 실행되든 위로 올라가면서 2025-LLM-Project 폴더를 찾아내는 GPS 함수"

# 코드 길이는 짧지만 os.getcwd, os.path.abspath, os.path.basename, os.path.dirname이라는 4가지 경로 조작 함수를 모두 활용하는, 파이썬 파일시스템 다루기의 좋은 압축 예제이기도 합니다.
# 궁금한 부분(예: os.path.basename vs os.path.dirname의 정확한 차이, 윈도우와 리눅스 경로 구분자 처리 등) 있으면 더 자세히 설명해드릴게요!

# 📦 1) Import
import os  # 운영체제 관련 모듈 - 경로 조작 함수들이 들어있음

# 🔍 2) get_project_root_dir 함수 - 프로젝트 루트 찾기
def get_project_root_dir(project_name: str = "2025-LLM-Project") -> str:
    """
    현재 파일 또는 실행 위치에서 프로젝트 루트 디렉토리를 탐색합니다.
    """
    # 현재 작업 디렉토리(Current Working Directory)의 절대 경로를 가져옴
    # os.getcwd(): "지금 어디서 파이썬을 실행 중인가?"의 경로
    # os.path.abspath(): 상대경로를 절대경로로 변환 (이미 절대경로면 그대로)
    cwd = os.path.abspath(os.getcwd())
    
    # 루프 조건: 현재 경로와 그 부모 경로가 다른 동안 계속 반복
    # → 파일시스템 최상단(루트)에 도달하면 cwd == dirname(cwd)가 되어 종료
    while cwd != os.path.dirname(cwd):
        # os.path.basename(): 경로의 "마지막 폴더 이름"만 추출
        # 예: "/home/user/2025-LLM-Project/src" → "src"
        # 현재 폴더 이름이 찾으려는 프로젝트 이름과 같으면 → 찾음!
        if os.path.basename(cwd) == project_name:
            return cwd  # 즉시 그 경로를 반환하고 함수 종료
        
        # 못 찾았으면 한 단계 위(부모 디렉토리)로 올라감
        # os.path.dirname(): 경로에서 마지막 폴더를 뗀 부분 반환
        # 예: "/home/user/2025-LLM-Project/src" → "/home/user/2025-LLM-Project"
        cwd = os.path.dirname(cwd)
    
    # 루프가 끝까지 갔는데도 못 찾았으면 (= 시스템 루트까지 갔는데도 없으면)
    # 에러를 발생시켜 프로그램을 중단
    raise RuntimeError(f"❌ 루트 디렉토리({project_name})를 찾을 수 없습니다.")