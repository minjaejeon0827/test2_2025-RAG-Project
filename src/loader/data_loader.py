# 소스코드 분석 (2026.05.20 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 이 파일은 RAG 시스템의 첫 단계인 "데이터 로딩 및 전처리" 부분이에요. PDF/HWP 파일에서 텍스트를 뽑아내고, 사용자 질문과 관련 있는 문서를 메타데이터로 빠르게 찾아주는 역할을 합니다.
# 💡 왜 코사인 유사도 함수가 두 개? 임베딩 모델 종류에 따라 출력 형태(텐서/리스트)가 달라서, 둘 다 준비해두고 상황에 맞게 골라 씀.

# 🎯 RAG 파이프라인에서의 위치
# README 구조도와 매핑하면 이 파일은 src/loader/ 에 해당해요:

# data_process → 원본 PDF/HWP에서 텍스트 추출 → 벡터DB 구축 전 단계
# extract_text_from_pdf → PDF의 일반 텍스트 + 이미지(OCR)까지 모두 캡처
# retrieve_top_documents_from_metadata → 사용자 질문 시 후보 문서를 빠르게 좁힘 (벡터DB 검색 전에 한 번 더 걸러주는 단계)

# 추가로 궁금한 부분(예: apply vs iterrows 차이, cos_sim이 정확히 어떤 수식인지, loc/iloc의 차이 등) 있으면 더 깊이 설명해드릴게요!

# 📦 1) Import 부분
import os                # 운영체제 관련 (파일 경로 처리, 존재 확인 등)
import easyocr           # 이미지 속 글자를 인식하는 OCR 라이브러리 (한국어 지원 좋음)
import fitz              # PyMuPDF의 별칭 - PDF 파일을 다루는 라이브러리 (텍스트/이미지 추출)
import numpy as np       # 수치 연산용 라이브러리 (이미지 배열, 유사도 계산 등에 사용)
import pandas as pd      # 표 형태 데이터(DataFrame) 처리 라이브러리

from pathlib import Path  # 파일 경로를 객체로 다루는 모듈 (os.path보다 현대적)
from PIL import Image     # Python Imaging Library - 이미지 데이터 변환에 사용
from tqdm import tqdm     # 반복문에 진행률 바를 표시해주는 라이브러리

from langchain_teddynote.document_loaders import HWPLoader
# 한국어 한글(HWP) 파일을 LangChain Document 형식으로 로딩하는 도구

from tabulate import tabulate
# 리스트/딕셔너리 데이터를 예쁜 표 형태로 출력해주는 라이브러리

from sklearn.metrics.pairwise import cosine_similarity
# scikit-learn의 코사인 유사도 계산 함수 (벡터 간 유사도 측정)

from sentence_transformers.util import cos_sim
# sentence-transformers의 코사인 유사도 계산 함수 (PyTorch 텐서 지원, GPU 가속 가능)

# 🔍 2) safe_ocr 함수 - 이미지에서 텍스트 추출
# 💡 safe_라는 접두사는 에러가 나도 프로그램 전체가 죽지 않도록 try/except로 감쌌다는 뜻이에요.
def safe_ocr(img_array: np.ndarray, ocr_reader: easyocr.Reader) -> str:
    """
    EasyOCR를 이용해 이미지 배열에서 텍스트를 추출합니다.
    """
    try:
        # OCR 실행: 이미지 배열을 받아 인식된 텍스트 리스트를 반환
        # detail=0 → 좌표/신뢰도 없이 텍스트만 반환 (간결한 형식)
        result = ocr_reader.readtext(img_array, detail=0)
        
        # 결과가 리스트가 아니면 (예상치 못한 반환) 빈 문자열 반환
        # → 방어적 코딩으로 다음 처리에서 에러 안 나게 함
        if not isinstance(result, list):
            return ""
        
        # 리스트의 각 텍스트 조각을 줄바꿈으로 이어붙여 하나의 문자열로 반환
        return "\n".join(result)
    except Exception as e:
        # OCR 처리 중 어떤 예외든 발생하면 RuntimeError로 감싸서 다시 발생
        # → 어느 함수에서 문제가 났는지 추적하기 쉬워짐
        raise RuntimeError(f"❌ [Runtime] (data_loader.safe_ocr) OCR 처리 실패: {e}")
    
# 📄 3) extract_text_from_pdf 함수 - PDF에서 텍스트 추출
# 💡 핵심 트릭: extract_text_from_pdf.reader = ...는 함수 객체에 속성을 붙이는 파이썬의 특이한 방식. 여러 번 호출돼도 OCR 모델은 한 번만 로딩되도록 하는 캐싱 기법입니다.
def extract_text_from_pdf(pdf_path: Path, apply_ocr: bool = True) -> str:
    """
    PDF 파일에서 텍스트를 추출하고, 필요시 OCR 결과도 병합합니다.
    """
    # 파일이 실제로 존재하는지 먼저 확인 (없으면 즉시 에러)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"❌ [FileNotFound] (data_loader.extract_text_from_pdf.pdf_path) PDF 파일을 찾을 수 없습니다: {pdf_path}"
        )

    # 모든 페이지의 텍스트를 누적할 빈 문자열
    full_text = ""
    try:
        # fitz.open으로 PDF 열기 - with 구문으로 자동 닫기 보장
        with fitz.open(pdf_path) as doc:
            # 페이지 순회 + tqdm으로 진행률 표시 (파일명을 라벨로)
            # enumerate로 페이지 번호(0부터)도 함께 얻음
            for page_num, page in enumerate(tqdm(doc, desc=f"{pdf_path.name}")):
                try:
                    # 페이지에서 텍스트 레이어 추출 (PDF에 내장된 글자)
                    # 이미지 기반 PDF면 빈 문자열이 나올 수 있음
                    page_text = page.get_text()
                    full_text += page_text  # 누적

                    # OCR 옵션이 켜져있으면 추가로 이미지 OCR 수행
                    if apply_ocr:
                        # 페이지를 300 DPI 고해상도 이미지로 렌더링
                        # DPI가 높을수록 OCR 정확도 ↑, 속도 ↓
                        pix = page.get_pixmap(dpi=300)
                        
                        # PyMuPDF의 픽셀 데이터를 PIL Image 객체로 변환
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # OCR 리더를 함수의 속성으로 한 번만 생성 (싱글톤 패턴)
                        # → 매 페이지마다 새로 만들면 메모리 낭비 + 느림
                        if not hasattr(extract_text_from_pdf, "reader"):
                            import torch
                            # GPU 사용 가능하면 GPU로, 아니면 CPU로 OCR 수행
                            gpu_available = torch.cuda.is_available()
                            # 한국어('ko')와 영어('en') 인식 가능한 리더 생성
                            extract_text_from_pdf.reader = easyocr.Reader(['ko', 'en'], gpu=gpu_available)
                        
                        # PIL Image를 NumPy 배열로 변환 후 OCR 수행
                        ocr_text = safe_ocr(np.array(img), extract_text_from_pdf.reader)
                        
                        # OCR 결과가 공백이 아니면 (실제 텍스트가 인식됐으면)
                        # 페이지 번호를 표시하면서 full_text에 덧붙임
                        if ocr_text.strip():
                            full_text += f"\n[OCR p.{page_num + 1}]\n{ocr_text}"
                except Exception as e:
                    # 한 페이지 처리 실패 → 경고만 출력하고 다음 페이지 진행
                    # → 한 페이지 깨졌다고 전체 PDF를 포기하지 않음
                    print(f"⚠️ [Warning] (data_loader.extract_text_from_pdf) 페이지 {page_num + 1} 처리 중 오류: {e}")
    except Exception as e:
        # PDF 자체를 열지 못하는 등 심각한 오류는 RuntimeError로 던짐
        raise RuntimeError(f"❌ [Runtime] (data_loader.extract_text_from_pdf) PDF 파일 처리 오류: {e}")

    return full_text  # 전체 페이지 텍스트(+OCR) 누적 결과 반환

# 🎯 4) retrieve_top_documents_from_metadata 함수 - 메타데이터 기반 문서 검색
# 💡 2단계 검색의 첫 단계입니다. 수백 개 RFP 중 메타데이터(제목/요약/기관명)만 가지고 빠르게 후보 5개로 좁힌 다음, 그 5개의 본문을 더 깊이 분석하는 구조예요.
def retrieve_top_documents_from_metadata(
    query, csv_path, embeddings, chat_history, top_k=5
):
    """
    사용자 질문과 메타데이터를 기반으로 유사도 검색을 수행합니다.
    """
    # ===== 입력 검증 단계 =====
    # 임베딩 모델이 안 들어왔으면 에러
    if embeddings is None:
        raise ValueError("❌ [Value] (data_loader.retrieve_top_documents_from_metadata) embedder 인자 누락")

    # 대화 이력이 안 들어왔으면 에러 (질문 맥락 보강에 필요)
    if chat_history is None:
        raise ValueError("❌ [Value] (data_loader.retrieve_top_documents_from_metadata) chat_history 인자 누락")

    # CSV 파일이 실제 존재하는지 확인
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ [FileNotFound] (data_loader.retrieve_top_documents_from_metadata) 파일을 찾을 수 없습니다: {csv_path}"
        )

    # ===== CSV 로딩 단계 =====
    try:
        # 메타데이터 CSV를 DataFrame으로 읽음
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(
            f"❌ [Value] (data_loader.retrieve_top_documents_from_metadata) CSV 파일 로딩 실패: {e}"
        )

    # CSV에 반드시 있어야 하는 컬럼 목록
    required_columns = ["사업명", "발주 기관", "사업 요약", "파일명"]
    for col in required_columns:
        # 하나라도 없으면 KeyError 발생
        if col not in df.columns:
            raise KeyError(
                f"❌ [Key] (data_loader.retrieve_top_documents_from_metadata) '{col}' 열이 CSV에 존재하지 않습니다."
            )

    # ===== 임베딩용 텍스트 생성 단계 =====
    # 각 행의 메타데이터를 하나의 문자열로 합치는 함수
    # → 이 통합 텍스트가 임베딩(벡터화)의 입력이 됨
    # chat_history를 앞에 붙여 대화 맥락까지 반영
    def make_embedding_text(row):
        return f"{chat_history} {row['파일명']} {row['사업 요약']} {row['사업명']} {row['발주 기관']}"

    try:
        # DataFrame의 각 행에 함수 적용 → "임베딩텍스트" 컬럼 생성
        # axis=1 → 행 단위로 함수 적용 (axis=0이면 열 단위)
        df["임베딩텍스트"] = df.apply(make_embedding_text, axis=1)
    except Exception as e:
        raise RuntimeError(f"❌ (data_loader.retrieve_top_documents_from_metadata) 임베딩 텍스트 생성 중 오류: {e}")

    # 모든 행의 임베딩 텍스트를 리스트로 변환
    doc_texts = df["임베딩텍스트"].tolist()

    # ===== 임베딩 및 유사도 계산 단계 =====
    # 임베딩 모델 종류에 따라 다른 방식으로 처리 (분기 처리)
    if hasattr(embeddings, "encode"):
        # SentenceTransformer 계열 (예: KoE5)
        # encode 메서드가 있으면 이 분기
        doc_embeddings = embeddings.encode(doc_texts, convert_to_tensor=True)  # 모든 문서 벡터화
        query_embedding = embeddings.encode(query, convert_to_tensor=True)    # 질문 벡터화
        # PyTorch 텐서 기반 코사인 유사도 → CPU로 옮기고 NumPy 배열로 변환
        similarities = cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
    else:
        # LangChain Embeddings 계열 (예: OpenAI text-embedding-3-large)
        # embed_documents / embed_query 메서드 사용
        doc_embeddings = embeddings.embed_documents(doc_texts)  # 문서들 벡터화 (리스트의 리스트)
        query_embedding = embeddings.embed_query(query)         # 질문 벡터화 (단일 리스트)
        # scikit-learn으로 코사인 유사도 계산
        # np.array([query_embedding]) → 2차원 배열로 변환 (1xN)
        similarities = cosine_similarity(
            np.array([query_embedding]), np.array(doc_embeddings)
        )[0]  # 결과의 첫 행만 추출

    # ===== 상위 K개 추출 단계 =====
    # argsort: 유사도를 정렬했을 때의 "인덱스" 반환 (오름차순)
    # [::-1]: 역순 (내림차순으로 만들기)
    # [:top_k]: 상위 K개만 자르기
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    try:
        # 상위 K개 행을 추출 (.copy()로 원본 보호)
        top_docs = df.iloc[top_k_indices].copy()
        # 유사도 컬럼을 새로 추가
        top_docs["유사도"] = similarities[top_k_indices]
    except Exception as e:
        raise RuntimeError(
            f"❌ [Runtime] (data_loader.retrieve_top_documents_from_metadata) 결과 DataFrame 생성 실패: {e}"
        )

    # ===== 결과 출력 단계 (디버깅용 표 출력) =====
    # 인덱스, 파일명, 유사도를 표 형태로 정리
    # f"{값:.4f}" → 소수점 4자리까지 표시
    table = [[idx, row["파일명"], f"{row['유사도']:.4f}"] for idx, row in top_docs.iterrows()]
    
    # tabulate로 GitHub 스타일 마크다운 표 만들기
    output = tabulate(table, headers=["IDX", "파일명", "유사도"], tablefmt="github")
    
    # 각 줄 앞에 공백 4칸 들여쓰기해서 보기 좋게 출력
    print("\n".join("    " + line for line in output.splitlines()))

    return top_docs  # 상위 K개 문서 DataFrame 반환

# 🗂️ 5) data_process 함수 - 파일 일괄 처리
def data_process(df: pd.DataFrame, config: dict, apply_ocr: bool = True, file_type: str = "all") -> pd.DataFrame:
    """
    주어진 파일 목록(DataFrame)을 기반으로 HWP 또는 PDF 파일을 읽어 텍스트를 추출합니다.
    """
    # config에서 프로젝트 루트 경로를 가져옴
    base_dir = config['settings']['project_root']
    # 실제 파일들이 있는 폴더 경로 조합: <루트>/data/files
    file_root = os.path.join(base_dir, "data", "files")

    # ===== 파일 타입 필터링 단계 =====
    if file_type in ["hwp", "pdf"]:
        # 파일명이 .hwp나 .pdf로 끝나는 행만 골라냄
        # str.lower()로 대소문자 무시 (PDF/pdf 둘 다 매칭)
        # mask는 True/False로 구성된 시리즈
        mask = df["파일명"].str.lower().str.endswith(f".{file_type}")
        filtered_df = df[mask].copy()  # .copy()로 원본과 분리
    elif file_type == "all":
        # 전체 다 처리할 때는 그냥 복사
        filtered_df = df.copy()

    # 텍스트를 저장할 새 컬럼을 None으로 초기화
    filtered_df["full_text"] = None

    # ===== 각 파일 순회 처리 단계 =====
    for file_name in filtered_df["파일명"]:
        # 실제 파일 경로 조합
        file_path = os.path.join(file_root, file_name)
        try:
            # 파일이 실제 존재하지 않으면 에러
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"❌ [FileNotFound] (data_loader.data_process.path) 파일이 존재하지 않습니다: {file_path}"
                )

            # === HWP 파일 처리 ===
            if file_name.lower().endswith(".hwp") and file_type in ["hwp", "all"]:
                loader = HWPLoader(file_path)  # HWP 로더 인스턴스 생성
                docs = loader.load()           # 문서 로딩 (LangChain Document 리스트 반환)
                
                # 문서가 존재하고, 첫 번째 문서의 내용이 문자열이면
                if docs and isinstance(docs[0].page_content, str):
                    # 해당 파일명에 해당하는 행의 full_text 컬럼에 내용 저장
                    # .loc[조건, 컬럼명] = 값 → DataFrame의 특정 셀 수정
                    filtered_df.loc[filtered_df["파일명"] == file_name, "full_text"] = docs[0].page_content
                else:
                    # 내용이 비어있으면 경고만 출력하고 그냥 None으로 둠
                    print(f"⚠️ [Warning] (data_loader.data_process.hwp) HWP 파일 무시됨 (내용 없음): {file_name}")

            # === PDF 파일 처리 ===
            elif file_name.lower().endswith(".pdf") and file_type in ["pdf", "all"]:
                # 앞서 만든 extract_text_from_pdf 함수 호출 (OCR 옵션 함께 전달)
                text = extract_text_from_pdf(Path(file_path), apply_ocr=apply_ocr)
                # 해당 행의 full_text 컬럼에 추출된 텍스트 저장
                filtered_df.loc[filtered_df["파일명"] == file_name, "full_text"] = text

            # === 지원하지 않는 형식 ===
            else:
                print(f"⚠️ [Warning] (data_loader.data_process) 지원하지 않는 파일 형식입니다: {file_name}")

        except Exception as e:
            # 한 파일이라도 실패하면 RuntimeError로 즉시 중단
            # (페이지 단위 처리와 달리 파일 단위 오류는 심각하게 다룸)
            raise RuntimeError(
                f"❌ [Runtime] (data_loader.data_process) 파일 처리 오류 ({file_name}): {e}"
            )

    # ===== 누락된 파일 보고 단계 =====
    # full_text가 NaN(None)인 행들의 파일명 추출
    # → 처리는 됐지만 내용이 비어있는 파일 목록
    empty_files = filtered_df[filtered_df["full_text"].isna()]["파일명"].tolist()
    if empty_files:
        print(f"⚠️ [Warning] (data_loader.data_process) 다음 파일은 내용이 없습니다: {', '.join(empty_files)}")

    # 인덱스를 0부터 다시 매겨서 깔끔하게 반환
    # drop=True → 기존 인덱스를 컬럼으로 보존하지 않고 버림
    return filtered_df.reset_index(drop=True)
