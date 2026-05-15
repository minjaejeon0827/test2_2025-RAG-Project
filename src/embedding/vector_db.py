# 소스코드 분석 (2026.05.15 minjae)
# 참고: https://gemini.google.com/app/7192578f5d43d984?hl=ko

# 안녕하세요! 올려주신 RFPilot 프로젝트의 README 파일과 vector_db.py 코드를 완벽하게 분석했습니다.

# README를 읽어보니, 여러분의 팀이 공공기관 제안요청서(RFP)라는 아주 실무적이고 방대한 데이터를 다루기 위해 얼마나 치열하게 고민하고 훌륭한 시스템(RAG)을 설계했는지 한눈에 보이네요. 5명의 팀원분들이 각자의 역할을 맡아 이렇게 체계적인 폴더 구조와 파이프라인을 완성하셨다니 정말 대단합니다!

# 이번에 보여주신 vector_db.py 파일은 이전 시간에 만든 '도서관(DB)' 시스템의 연장선에 있는 '실무 담당 사서' 역할을 하는 아주 핵심적인 코드입니다. 글자를 숫자로 바꾸는 기계를 세팅하고, 수많은 문서를 무너지지 않게 책장에 차곡차곡 꽂아 넣는 실제 작업 지시서라고 볼 수 있어요.

# 코드가 길고 에러(예외 처리)를 대비한 꼼꼼한 코드들이 많아 복잡해 보일 수 있지만, 중학생 친구에게 설명하듯 아주 쉽고 친절하게 한 줄 한 줄 주석을 달아드릴게요. 편안하게 읽어보세요! 😊

# 💡 핵심 요약!
# 이 파일의 가장 훌륭한 점은 '안정성'입니다. 단순히 문서를 데이터베이스에 넣는 기능만 만든 것이 아니라, 만약 수천 장짜리 제안서가 들어왔을 때 컴퓨터가 멈추지 않도록 add_docs_in_batch라는 함수를 만들어서 128개씩 안전하게 나눠서 처리하도록 한 부분이 아주 인상적입니다. 또한 중간에 진행 상태를 볼 수 있게 tqdm을 달아둔 것도 사용자 경험(UX) 측면에서 100점 만점짜리 설계네요!

# ==========================================
# 1. 외부 도구(라이브러리) 가져오기
# ==========================================
import os        # 컴퓨터의 파일이나 폴더 경로를 다루는 도구예요.
import faiss     # 메타(Meta)에서 만든 아주 빠른 숫자 도서관(벡터 DB) 도구예요.
import shutil    # 폴더를 통째로 지울 때 쓰는 강력한 청소부예요.

from tqdm import tqdm # "지금 책 100권 중 50권 꽂았어!" 하고 진행률 바(게이지)를 보여주는 도구예요.
from typing import List, Union, Optional # "이건 리스트야", "이건 선택사항이야" 하고 힌트를 줘요.

from langsmith import traceable # 프로그램이 어디서 헤매는지 발자국을 추적하는 기능이에요.
from langchain_chroma import Chroma # '크로마'라는 또 다른 종류의 숫자 도서관이에요.
from langchain.schema import Document # 글자를 담는 예쁜 상자(문서 객체)예요.
from langchain_openai import OpenAIEmbeddings # 오픈AI의 '글자->숫자 번역기'예요.
from langchain_community.vectorstores import FAISS # FAISS 도서관을 더 쉽게 쓰게 해주는 도구예요.
from langchain.vectorstores.base import VectorStore # 도서관들의 기본 뼈대(설계도)예요.
from langchain_huggingface import HuggingFaceEmbeddings # 무료 AI 창고(허깅페이스)의 번역기예요.
from langchain_community.docstore.in_memory import InMemoryDocstore # 문서를 임시로 컴퓨터 기억장치(메모리)에 올려두는 도구예요.

# ==========================================
# [함수 1] 글자를 숫자로 바꾸는 '번역기' 준비하기
# ==========================================
def generate_embedding(embed_model_name: str) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
    """
    어떤 번역기(임베딩 모델)를 쓸지 이름을 주면, 그 기계를 짠! 하고 준비해 주는 함수예요.
    """
    try:
        if embed_model_name == "openai": # 만약 "openai" 번역기를 쓰겠다고 하면?
            api_key = os.getenv("OPENAI_API_KEY") # 컴퓨터 안에 숨겨둔 비밀번호(API 키)를 찾아와요.
            if not api_key: # 비밀번호가 없으면?
                raise ValueError("❌ [Value] (vector_db.generate_embedding.api_key) OPENAI_API_KEY 누락") # 에러를 내며 멈춰요!
            # 비밀번호가 맞다면 최신 똑똑한 번역기("text-embedding-3-large")를 준비해서 돌려줍니다.
            return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        else:
            # "openai"가 아니면 허깅페이스에 있는 무료 번역기를 이름대로 찾아와서 준비해 줘요.
            return HuggingFaceEmbeddings(model_name=embed_model_name)
    except Exception as e:
        # 기계를 준비하다가 뭔가 고장나면 에러 이유를 알려줘요.
        raise ValueError(f"❌ [Value] (vector_db.generate_embedding.init) 임베딩 모델 초기화 실패 원인: {e}")


# ==========================================
# [함수 2] 새 도서관(DB) 만들고 책(문서) 꽂아 넣기
# ==========================================
@traceable(name="generate_vector_db")
def generate_vector_db(
    all_chunks: List[Document], # 잘게 자른 제안서 문서 조각들 (책들)
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], # 방금 위에서 만든 번역기
    index_name: str, # 도서관의 이름표
    db_type: str = "faiss", # 도서관 종류 (기본값은 faiss)
    is_save: bool = True, # 다 꽂고 컴퓨터에 저장할지 여부
    output_path: str = "" # 도서관이 지어질 주소
) -> Union[FAISS, Chroma]:
    
    # --- 1단계: 재료 검사 ---
    # 책이 없거나 모양이 이상하면 돌려보내요.
    if not all_chunks or not isinstance(all_chunks, list):
        raise ValueError("❌ [Value] 비어 있거나 잘못된 Document 리스트")
    # 번역기가 고장났거나 이상한 거면 돌려보내요.
    if not isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
        raise ValueError("❌ [Value] 잘못된 임베딩 객체")
    # 이름표나 주소가 없어도 돌려보내요.
    if not index_name:
        raise ValueError("❌ [Value] 빈 index_name 인자")
    if not output_path:
        raise ValueError("❌ [Value] 빈 output_path 인자")

    db_type = db_type.lower() # 이름을 소문자로 깔끔하게 맞춰줘요.
    print(f"📌 [Info] 임베딩 모델: {embeddings.__class__.__name__}") # 어떤 번역기를 쓰는지 터미널에 알려줘요.

    try:
        # --- 2단계: 책장 크기(차원) 재기 ---
        # "hello world"라는 말을 번역기에 넣어보고, 숫자가 몇 개(차원)로 나오는지 길이를 재요. (책장 칸 수를 정하는 거예요)
        dimension = 3072 if isinstance(embeddings, OpenAIEmbeddings) else len(embeddings.embed_query("hello world"))
    except Exception as e:
        raise ValueError(f"❌ [Value] 임베딩 차원 계산 실패 원인: {e}")

    try:
        os.makedirs(output_path, exist_ok=True) # 도서관을 지을 폴더를 만들어요.

        # --- 3단계: 도서관 종류에 맞춰 책 꽂기 ---
        if db_type == "faiss":
            print(f"📌 [Info] 벡터 DB 유형: {db_type}")
            # 빈 FAISS 책장을 만들어요. (방금 잰 칸 수(dimension)에 맞춰서요)
            vector_store = FAISS(
                embedding_function=embeddings,
                index=faiss.IndexFlatL2(dimension),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            # 카트에 책을 나눠 담아서 조심조심 책장에 꽂아 넣어요. (아래쪽에 있는 함수를 불러와요!)
            vector_store = add_docs_in_batch(vector_store, all_chunks)
            
            if is_save: # 다 꽂았으면 컴퓨터에 파일로 예쁘게 저장해요.
                vector_store.save_local(folder_path=output_path, index_name=index_name)
                print("✅ [Success] FAISS 벡터 DB 저장 완료")

        elif db_type == "chroma":
            print(f"📌 [Info] 벡터 DB 유형: {db_type}")
            chroma_path = os.path.join(output_path, index_name)
            
            # 만약 예전에 쓰다 만 낡은 도서관이 있으면 청소부(shutil)를 불러서 싹 밀어버려요.
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
                print(f"⚠️ [Warning] 기존 Chroma DB 제거 완료")

            # 새 Chroma 책장을 만들어요.
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_path,
                collection_name="chroma_db",
            )
            # 책을 꽂아요. (Chroma는 꽂으면 자동으로 저장돼요)
            vector_store = add_docs_in_batch(vector_store, all_chunks)
            print("✅ [Success] Chroma 벡터 DB 저장 완료")

        else:
            # 이상한 도서관 이름을 대면 화를 냅니다.
            raise ValueError("❌ [Value] 지원하지 않는 벡터 DB 타입 ('faiss' 또는 'chroma'만 가능)")

        return vector_store # 완성된 도서관을 반환해요!

    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] 벡터 DB 생성 실패 원인: {e}")


# ==========================================
# [함수 3] 이미 지어진 도서관(DB) 문 열고 들어가기
# ==========================================
@traceable(name="load_vector_db")
def load_vector_db(
    path: str, # 도서관 주소
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], # 번역기
    index_name: str, # 도서관 이름표
    db_type: str = "faiss" # 도서관 종류
) -> Union[FAISS, Chroma]:
    """
    새로 만들 필요 없이, 어제 만들어둔 도서관을 그대로 불러오는 함수예요.
    """
    # 주소에 갔는데 도서관이 없으면 에러!
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ [FileNotFound] 존재하지 않는 벡터 DB 경로: {path}")

    if not isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
        raise ValueError("❌ [Value] 잘못된 임베딩 객체")

    if not index_name:
        raise ValueError("❌ [Value] 빈 index_name 인자")

    db_type = db_type.lower()

    try:
        if db_type == "faiss":
            # FAISS 도서관 문을 열어요. (allow_dangerous_deserialization=True는 파이썬 파일을 믿고 연다는 뜻이에요)
            return FAISS.load_local(
                folder_path=path,
                index_name=index_name,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        elif db_type == "chroma":
            # Chroma 도서관 문을 열어요.
            chroma_path = os.path.join(path, index_name)
            return Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_path,
                collection_name="chroma_db",
            )
        else:
            raise ValueError(f"❌ [Value] 지원하지 않는 벡터 DB 타입: {db_type}")

    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] 벡터 DB 로드 실패 원인: {e}")


# ==========================================
# [함수 4] 책을 '수레(Batch)'에 나눠서 조심조심 꽂기
# ==========================================
def add_docs_in_batch(
    vector_store: VectorStore, # 채워 넣을 책장(도서관)
    chunks: Optional[List[Document]], # 꽂아야 할 수많은 책(문서)들
    batch_size: int = 128 # 한 수레에 담을 책의 개수 (기본 128개)
) -> VectorStore:
    """
    수천 개의 문서를 한 번에 꽂으면 컴퓨터가 무거워서 기절할 수 있어요.
    그래서 128개씩 카트에 나눠 담아서 조금씩 꽂아주는 아주 똑똑한 안전 함수예요!
    """
    if not chunks or not isinstance(chunks, list):
        raise ValueError("❌ [Value] 비어 있거나 잘못된 Document 리스트")

    if batch_size <= 0: # 카트 크기가 0이거나 마이너스면 말이 안 되죠!
        raise ValueError("❌ [Value] batch_size는 1 이상이어야 함")

    total = len(chunks) # 총 꽂아야 할 책이 몇 권인지 세어봐요.
    
    # 화면에 진행 상황을 예쁘게 보여주는 로딩 바(게이지)를 만들어요.
    pbar = tqdm(
        range(0, total, batch_size), # 0부터 끝까지 카트 크기만큼 껑충껑충 뛰면서 반복해요. (예: 0, 128, 256...)
        desc=f"📌 [Info] {vector_store.__class__.__name__} 인덱싱 진행 중", # 로딩 바 옆에 띄울 설명글이에요.
        unit="batch",
    )

    try:
        # 카트에 나눠 담고 나르는 진짜 작업 시작!
        for i in pbar:
            batch = chunks[i:i + batch_size] # 딱 이번에 나를 128권만 잘라서 챙겨요.
            vector_store.add_documents(batch) # 책장에 꽂아 넣어요!

            # 로딩 바 옆에 "지금 총 1000권 중에 256권 꽂았어!" 하고 숫자를 업데이트해 줘요.
            end_idx = min(i + batch_size, total)
            pbar.set_postfix_str(f"진행 {end_idx} / {total}")

        return vector_store # 안전하게 다 꽂은 책장을 반환해요.

    except Exception as e:
        # 꽂다가 책장을 엎어버렸다면(?) 에러를 내요.
        raise RuntimeError(f"❌ [Runtime] 문서 배치 삽입 실패 원인: {e}")