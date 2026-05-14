# 소스코드 분석 (2026.05.14 minjae)
# 참고: https://gemini.google.com/app/7192578f5d43d984?hl=ko

# 안녕하세요! 올려주신 RFPilot 프로젝트의 README 파일과 embedding_main.py 코드를 꼼꼼하게 잘 읽어보았습니다.

# 이전의 chatbot_original.py가 손님을 맞이하는 '식당 홀'이고, main.py가 주문을 받는 '주방'이었다면, 이번에 보여주신 embedding_main.py는 '재료(문서)를 아주 잘게 썰어서 찾기 쉽게 냉장고(데이터베이스)에 착착 정리해두는 과정'이라고 생각하시면 됩니다.

# 코드를 보기 전에 먼저 아주 중요한 개념 두 가지만 쉽게 짚고 넘어갈게요!

# 임베딩(Embedding): 컴퓨터는 한글이나 영어를 그대로 이해하지 못해요. 그래서 컴퓨터가 이해할 수 있는 '숫자(벡터)'로 글자를 변환해 주는 마법 같은 과정입니다.

# 벡터 DB(Vector DB): 숫자로 변환된 글자들을 저장해두는 아주 똑똑한 '도서관'이에요. 나중에 질문이 들어오면, 질문과 가장 비슷한 의미를 가진 문서 조각을 여기서 순식간에 찾아냅니다. (이 프로젝트에서는 FAISS와 Chroma라는 두 가지 종류의 도서관을 쓰네요!)

# 자, 그럼 이 파일이 어떻게 똑똑한 도서관을 만들고 관리하는지, 중학생 친구에게 설명하듯 코드 한 줄 한 줄 친절하게 주석을 달아드릴게요. 천천히 따라와 보세요! 😊

# 💡 요약하자면!
# 이 코드는 "사용자가 입력한 설정값과 대화방 번호표를 조합해 데이터베이스의 고유한 이름을 짓고 ➡️ 예전에 지어둔 데이터베이스가 멀쩡히 있는지 검사한 다음 ➡️ 필요하면 문서를 변환해 새로 저장하거나, 아니면 기존 것을 똑똑하게 불러오는 역할"을 완벽하게 해내고 있습니다. 특히, 데이터가 짓다가 말아서 불완전할 경우 스스로 싹 지우고 다시 시작하는 꼼꼼함이 돋보이는 훌륭한 코드네요!

# 이 설명이 프로젝트의 전체적인 흐름을 이해하는 데 도움이 되었기를 바랍니다. 혹시 보시다가 궁금한 점이 생기면 언제든지 편하게 질문해 주세요!

# ==========================================
# 1. 외부 도구(라이브러리) 가져오기
# ==========================================
import os     # 컴퓨터 안의 폴더를 만들거나 경로를 찾을 때 쓰는 도구예요.
import shutil # 폴더 안에 내용물이 있어도 통째로 싹 지워주는 강력한 청소부 도구예요.

from typing import List, Union # "이건 리스트야", "이건 A 또는 B야"라고 파이썬에게 힌트를 주는 도구예요.
from langsmith import traceable # 프로그램이 어떻게 돌아가는지 발자국을 남겨서 추적(디버깅)하기 좋게 해줘요.
from langchain_chroma import Chroma # '크로마(Chroma)'라는 이름의 똑똑한 숫자 도서관(벡터 DB) 도구예요.
from langchain.schema import Document # 글자 덩어리들을 예쁘게 포장하는 '문서 상자' 도구예요.
from langchain_openai import OpenAIEmbeddings # OpenAI의 인공지능을 빌려 글자를 숫자로 바꾸는 번역기예요.
from langchain_community.vectorstores import FAISS # '파이스(FAISS)'라는 이름의 아주 빠른 숫자 도서관(벡터 DB) 도구예요.
from langchain_huggingface import HuggingFaceEmbeddings # 무료 인공지능 창고(허깅페이스)의 번역기예요.

# ==========================================
# 2. 우리 프로젝트 내부 도구 가져오기
# ==========================================
from src.utils.path import get_project_root_dir # 프로젝트의 맨 꼭대기(뿌리) 폴더를 찾아주는 우리 팀 함수예요.
from src.embedding.vector_db import generate_vector_db, load_vector_db # 도서관을 새로 짓거나, 만들어진 도서관을 열어주는 함수예요.


# ==========================================
# 3. 도서관 간판(인덱스 이름) 만들기 함수
# ==========================================
def generate_index_name(config: dict) -> str:
    """
    우리가 설정한 값들(문서 종류, 자르는 방식, 모델 이름 등)을 조합해서 
    도서관(벡터 DB)의 겹치지 않는 멋진 이름을 지어주는 함수예요.
    """
    # 설정 파일(config)에서 값들을 꺼내옵니다. (값이 없으면 기본값을 써요)
    data_type = config.get("data", {}).get("file_type", "all") # 파일 종류 (pdf, hwp, all 등)
    splitter = config.get("data", {}).get("splitter", "recursive") # 문서를 자르는 방법
    model = config.get("embedding", {}).get("embed_model", "default") # 사용한 번역기(임베딩) 모델 이름
    db_type = config.get("embedding", {}).get("db_type", "faiss") # 도서관 종류 (faiss 또는 chroma)

    # 모델 이름에 슬래시(/)나 하이픈(-), 띄어쓰기가 있으면 파일 이름으로 쓰기 나쁘니까
    # 깔끔하게 언더바(_)로 바꿔주는 청소 작업이에요.
    model_key = model.split("/")[-1] if "/" in model else model
    model_key = model_key.replace('-', '_').replace(' ', '_')

    # 만약 문서를 100개 꽉 채워서 쓴다면 이름에 100을 넣고, 아니면 뺍니다.
    # 예시 완성품: "pdf_100_recursive_openai_faiss" (이게 바로 도서관 이름표가 돼요!)
    if config.get('data', {}).get('top_k') == 100:
        return f"{data_type}_{config['data']['top_k']}_{splitter}_{model_key}_{db_type}"
    else:
        return f"{data_type}_{splitter}_{model_key}_{db_type}"


# ==========================================
# 4. 🌟 메인 핵심 함수: 임베딩 처리 및 도서관(DB) 관리
# ==========================================
@traceable(name="embedding_main") # 실행되는 과정을 추적하게 해주는 이름표를 붙여요.
def embedding_main(
    config: dict, # 전체 설정값 꾸러미
    chunks: List[Document], # 잘게 썰어놓은 문서 덩어리들 (상자에 담겨있어요)
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], # 글자를 숫자로 바꿀 번역기 (둘 중 하나)
    is_save: bool = False, # "이거 새로 만들어서 저장할까?" (True면 저장, False면 불러오기)
    session_id: str = None # 지금 대화하고 있는 방 번호표
) -> Union[FAISS, Chroma]: # 최종적으로 만들어진 FAISS나 Chroma 도서관을 내보낼 거예요.
    
    # ----------------------------------------
    # [1단계] 재료가 제대로 왔는지 깐깐하게 검사하기
    # ----------------------------------------
    # 문서 덩어리(chunks)가 리스트 형태가 아니거나, Document 상자에 안 담겨 있으면 에러를 내요!
    if not isinstance(chunks, list) or not all(isinstance(chunk, Document) for chunk in chunks):
        raise ValueError("❌ (embedding.embedding_main.chunks) chunks는 Document 객체의 리스트여야 합니다.")
    # 상자가 텅 비어있어도 에러!
    if len(chunks) == 0:
        raise ValueError("❌ (embedding.embedding_main.chunks) chunks 리스트가 비어 있음")
    
    # 번역기(embeddings)가 없거나, 우리가 아는 번역기가 아니면 에러!
    if embeddings is None or not isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
        raise ValueError("❌ (embedding.embedding_main.embeddings) 잘못된 embeddings 인자")

    # ----------------------------------------
    # [2단계] 도서관을 세울 땅(폴더 경로) 정하기
    # ----------------------------------------
    embed_config = config['embedding']
    db_type = embed_config['db_type'].lower() # 도서관 종류 이름을 소문자로 맞춰줘요.
    project_root = get_project_root_dir() # 프로젝트 맨 꼭대기 위치를 찾고
    # "최상단/data" 이런 식으로 도서관이 지어질 정확한 주소를 만들어요.
    vector_db_path = os.path.join(project_root, embed_config.get("vector_db_path", "data"))

    # 주소가 이상하면 에러!
    if not isinstance(vector_db_path, str) or vector_db_path.strip() == "":
        raise ValueError("❌ (embedding.embedding_main.vector_db_path) 잘못된 vector_db_path 경로")

    # 주소에 해당하는 폴더가 실제로 없으면? 컴퓨터한테 "폴더 좀 만들어줘!" 라고 명령해요. (exist_ok=True는 이미 있어도 에러 안 낸다는 뜻)
    os.makedirs(vector_db_path, exist_ok=True)
    
    # 아까 만든 함수로 도서관 이름표를 가져오고, 뒤에 대화방 번호표(session_id)를 붙여서 완성해요.
    index_name = generate_index_name(config)
    index_name = index_name + f"_{session_id}"

    # 이름표가 이상하면 에러!
    if not isinstance(index_name, str) or index_name.strip() == "":
        raise ValueError("❌ (embedding.embedding_main.index_name) 잘못된 index_name 생성")

    # ----------------------------------------
    # [3단계] 이전에 만들어둔 도서관이 있는지 확인하기
    # ----------------------------------------
    if db_type == "faiss":
        # FAISS 도서관은 보통 .faiss 파일과 .pkl 파일 두 개로 이루어져 있어요.
        faiss_file = os.path.join(vector_db_path, f"{index_name}.faiss")
        pkl_file = os.path.join(vector_db_path, f"{index_name}.pkl")
        # 두 파일이 모두 있다면 "어? 도서관이 이미 있네!"(True) 라고 기억해요.
        db_exists = os.path.exists(faiss_file) and os.path.exists(pkl_file)

    elif db_type == "chroma":
        # Chroma 도서관은 폴더 형태로 생겨요.
        chroma_dir = os.path.join(vector_db_path, index_name)
        sqlite_path = os.path.join(chroma_dir, "chroma.sqlite3")

        # 폴더 안에 필수 파일(sqlite3)이 있는지 확인해요.
        has_sqlite = os.path.exists(sqlite_path)
        # 내용물이 제대로 채워져 있는지 개수를 세어서 꼼꼼히 확인해요.
        has_index_dirs = any(
            os.path.isdir(os.path.join(chroma_dir, d)) and len(os.listdir(os.path.join(chroma_dir, d))) >= 4
            for d in os.listdir(chroma_dir)
            if os.path.isdir(os.path.join(chroma_dir, d))
        ) if os.path.exists(chroma_dir) else False

        # 둘 다 통과해야 "도서관이 완벽히 있다!"(True)고 인정해요.
        db_exists = has_sqlite and has_index_dirs

        # 만약 폴더는 있는데 내용물이 엉망이라면(짓다가 말았다면)?
        if os.path.exists(chroma_dir) and not db_exists:
            print("⚠️ 불완전한 Chroma 벡터 DB가 감지되어 삭제합니다.")
            shutil.rmtree(chroma_dir) # 불량품이므로 폴더를 싹 다 지워버리고 처음부터 다시 짓기로 해요. (이게 아까 가져온 청소부 도구예요!)
            db_exists = False

    else:
        # FAISS도 아니고 Chroma도 아니면 우리가 모르는 도서관이니까 에러!
        raise ValueError(f"❌ (embedding.embedding_main.db_type) 지원하지 않는 DB 타입입니다: {db_type}")

    # ----------------------------------------
    # [4단계] 진짜 도서관 세우기 (또는 문 열기)
    # ----------------------------------------
    if is_save:
        # 새로 저장하라는 명령(is_save=True)이 떨어졌다면,
        # 문서 덩어리들(chunks)을 가져다가 번역기(embeddings)로 숫자로 다 바꾸고 DB를 새로 만들어요!
        vector_store = generate_vector_db(chunks, embeddings, index_name, db_type, output_path=vector_db_path)
        print("✅ Vector DB 생성 완료")
    else:
        # 저장하라는 명령이 없다면, 이미 지어져 있는 도서관을 그냥 열어서 가져오기만 해요.
        vector_store = load_vector_db(vector_db_path, embeddings, index_name, db_type)
        print("✅ Vector DB 로드 완료")

    # ----------------------------------------
    # [5단계] 결과물 배달하기
    # ----------------------------------------
    # 완성된 도서관(또는 불러온 도서관)을 이 함수를 부른 쪽으로 짠! 하고 돌려줍니다.
    return vector_store