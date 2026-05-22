# 소스코드 분석 (2026.05.22 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3


# 🔄 전체 흐름 정리
# section 방식 기준으로 함수들이 협력하는 순서:
# data_chunking (총괄)
#    │
#    ├─ 1) extract_sections()      → 목차 패턴으로 섹션 분리
#    │                                (정규식 1차 → 실패 시 2차 → 실패 시 빈 리스트)
#    ├─ 2) clean_text()            → 각 섹션 본문 청소
#    ├─ 3) merge_short_chunks()    → 짧은 섹션끼리 병합 (500자 기준)
#    ├─ 4) refine_chunks_with_length_control() → 긴 섹션 재분할 (1000자 + 250 겹침)
#    └─ 5) Document 변환           → 메타데이터 붙여서 최종 청크 완성

# summarize_chunk_quality (검사)
#    └─ inspect_sample_chunks()    → 샘플 청크 눈으로 확인

# 🎯 한 줄 요약

# "긴 RFP 문서를 → 목차 기준으로 섹션 분리 → 청소 → 너무 짧으면 합치고 → 너무 길면 자르고 → 출처 정보를 붙여 검색 가능한 Document 조각으로 변환"

# 이렇게 만들어진 청크들이 loader_main을 통해 반환되어, 다음 단계인 임베딩 → 벡터DB 저장(src/embedding/)으로 넘어갑니다.
# 💡 이 파일의 설계가 똑똑한 이유
# 세 가지 분할 전략을 지원한다는 점이 핵심이에요. 단순 분할기(recursive, token)는 글자/토큰 수로만 기계적으로 자르지만, section 방식은 한국 공문서의 목차 구조를 이해하고 의미 단위로 자릅니다. RFP처럼 구조가 명확한 문서에 특화된, 도메인 지식이 반영된 설계예요.
# 궁금한 부분(예: 정규식의 (?:...)가 정확히 뭔지, RecursiveCharacterTextSplitter가 내부적으로 어떻게 자르는지, enumerate/defaultdict 같은 문법) 있으면 더 깊이 파고들어 설명해드릴게요!


# 드디어 앞서 loader_main.py가 호출하던 data_chunking과 summarize_chunk_quality의 실제 구현 파일이네요! 이 파일이 지금까지 본 것 중 가장 복잡한데, 핵심은 "긴 RFP 문서를 어떻게 의미 있는 작은 조각으로 똑똑하게 자를까"입니다. 정규식이 많이 나와서 그 부분을 특히 자세히 풀어드릴게요.
# 📦 1) Import 부분
import re          # 정규표현식(regular expression) - 문자열 패턴 매칭/치환에 사용
import numpy as np # 수치 연산 (청크 길이 통계 계산용)
import pandas as pd # 표 형태 데이터(DataFrame) 처리

from collections import defaultdict  # 기본값이 있는 딕셔너리 (없는 키 접근 시 자동 생성)
from typing import List              # 타입 힌트용
from langchain.schema import Document # 텍스트+메타데이터를 담는 LangChain 표준 객체
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
# RecursiveCharacterTextSplitter: 문단/문장/단어 순으로 자연스럽게 쪼개는 분할기
# TokenTextSplitter: 토큰 개수 기준으로 쪼개는 분할기

# 🧹 2) clean_text 함수 - 텍스트 정제
# 💡 re.sub(패턴, 바꿀것, 대상) = "대상에서 패턴에 맞는 부분을 바꿀것으로 치환". OCR 결과에 섞인 깨진 글자나 이상한 기호를 청소하는 단계예요.
def clean_text(text: str) -> str:
    """
    입력 문자열에서 불필요한 문자 및 공백을 정리합니다.
    """
    # 입력이 문자열이 아니면 에러 (예: 숫자, None 등이 들어오면 방어)
    if not isinstance(text, str):
        raise ValueError("❌ [Type] (splitter.clean_text) 문자열이 아닌 입력값")

    # "허용할 문자"를 정의하는 정규식 패턴
    # 맨 앞 ^ 는 "이것들을 제외한 나머지"라는 의미 (부정)
    # \uAC00-\uD7A3 : 한글 완성형 글자 (가~힣)
    # a-zA-Z0-9 : 영문 대소문자 + 숫자
    # \s : 공백류(스페이스, 탭, 줄바꿈)
    # .,:;!?'"()[]~-/ : 일반 문장부호
    # •※❍□ㅇ○①-⑳ : 문서에서 자주 쓰는 특수 기호/목차 기호
    # IVXLCDM : 로마 숫자용 알파벳
    allowed_pattern = r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,:;!?\'\"\(\)\[\]~\-/•※❍□ㅇ○①-⑳IVXLCDM]"
    
    # 위에서 정의한 "허용 목록에 없는 문자"를 전부 공백으로 치환
    # → 깨진 글자, 이상한 특수문자 등 제거
    text = re.sub(allowed_pattern, " ", text)
    
    # 연속된 공백(스페이스 여러 개, 줄바꿈 등)을 하나의 스페이스로 압축
    text = re.sub(r"\s+", " ", text)
    
    # 앞뒤 공백 제거 후 반환
    return text.strip()

# 📑 3) extract_sections 함수 - 목차 기반 섹션 분리 (핵심!)
# 💡 이 함수가 왜 중요한가? RFP 같은 공문서는 "1. 사업개요", "2. 사업내용", "가. 추진방향" 같은 목차 구조를 가져요. 단순히 글자 수로 자르면 문맥이 끊기지만, 목차 단위로 자르면 의미 단위로 깔끔하게 나뉩니다.
# 💡 fallback(폴백) 전략: 정교한 패턴(1차)으로 먼저 시도하고, 실패하면 단순 패턴(2차)으로 재시도. 둘 다 실패하면 빈 리스트. "최선 → 차선 → 포기"의 3단 안전망입니다.
def extract_sections(text: str) -> List[dict]:
    """
    다양한 목차 패턴을 기반으로 텍스트를 섹션 단위로 분리합니다.
    """
    # 두 개의 정규식 패턴을 순서대로 시도 (1차 실패 시 2차로 fallback)
    fallback_patterns = [
        # ===== 1차: 다양한 목차 형식을 모두 잡는 강화 패턴 =====
        re.compile(
            r"""
            (?:^|\n)[ \t]*               # 줄 시작 또는 줄바꿈 뒤, 공백/탭 허용
            (                            # 그룹 시작 (목차 번호 부분)
                (?:\d+(?:\.\d+)*[.)])       # 숫자 기반: 1.  1.1.  2.3.1) 등
                | (?:[가-힣]{1}[.)])         # 한글 한 글자: 가.  나.  다) 등
                | (?:\[\d+\])               # 대괄호 숫자: [1]  [2]
                | (?:\[\s*붙임\s*\d+\s*\])  # [붙임 1] 형식
                | (?:[①-⑳])                # 원문자: ① ② ... ⑳
                | (?:[○•※❍□ㅇ])            # 문서 특수 불릿 기호
                | (?:[IVXLCDM]{1,4}[.)])    # 로마숫자: I.  II.  IV) 등
            )                            # 그룹 끝
            [ \t]+                       # 번호 뒤 공백/탭 (필수)
            ([^\n]{2,100})               # 제목 텍스트: 줄바꿈 아닌 문자 2~100개
            """,
            re.MULTILINE | re.VERBOSE
            # MULTILINE: ^와 $가 각 줄마다 적용되게 함
            # VERBOSE: 정규식 안에 공백/주석 쓸 수 있게 함 (가독성용)
        ),
        # ===== 2차: 단순 숫자 목차만 인식하는 단순 패턴 =====
        # 1차에서 아무것도 못 찾았을 때 보험용
        re.compile(r"\n?(\d+(\.\d+)*\s?[.)]?\s+[^\n]{2,100})")
    ]

    # 두 패턴을 순서대로 시도
    for pattern in fallback_patterns:
        # finditer: 패턴에 맞는 모든 위치를 찾아 반복자로 반환
        matches = list(pattern.finditer(text))
        if matches:  # 매칭된 게 하나라도 있으면
            chunks = []
            for i in range(len(matches)):
                # 현재 목차의 시작 위치
                start = matches[i].start()
                # 다음 목차의 시작 위치 (= 현재 섹션의 끝)
                # 마지막 목차면 텍스트 끝까지
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                
                # 목차 줄 전체를 제목으로 사용
                title = matches[i].group().strip()
                # start부터 end까지의 텍스트를 섹션 본문으로
                content = text[start:end].strip()
                
                # 제목과 본문을 딕셔너리로 저장
                chunks.append({"title": title, "content": content})
            return chunks  # 첫 번째로 성공한 패턴 결과 반환

    # 두 패턴 모두 실패 시 경고 + 빈 리스트 반환
    print("⚠️ [Warning] (splitter.extract_sections) 섹션 패턴이 발견되지 않음")
    return []

# 4) merge_short_chunks 함수 - 짧은 청크 병합
# 💡 왜 병합하나? 목차로 자르다 보면 "다." 같은 한두 줄짜리 너무 작은 섹션이 생겨요. 이런 자투리는 검색 품질을 떨어뜨리므로, 다음에 나오는 제대로 된 섹션에 합쳐줍니다.
# ⚠️ 참고: 이 코드는 buffer가 마지막까지 비워지지 않으면(끝부분이 전부 짧은 청크면) 그 내용이 누락될 수 있는 구조예요. 작은 버그 포인트입니다.
def merge_short_chunks(chunks: List[dict], min_length: int = 500) -> List[dict]:
    """
    길이가 min_length 미만인 청크들을 인접한 청크에 병합하여 반환합니다.
    """
    merged = []   # 최종 결과 리스트
    buffer = ""   # 너무 짧은 청크들을 임시로 모아두는 공간
    
    for chunk in chunks:
        # 현재 청크 본문이 기준 길이(500자)보다 짧으면
        if len(chunk["content"]) < min_length:
            # 바로 결과에 넣지 않고 buffer에 쌓아둠
            buffer += " " + chunk["content"]
        else:
            # 충분히 긴 청크를 만났을 때
            if buffer:  # 그동안 모아둔 짧은 조각이 있으면
                # 그것들을 현재 청크 앞에 붙임
                chunk["content"] = buffer.strip() + " " + chunk["content"]
                buffer = ""  # buffer 비우기
            merged.append(chunk)  # 결과에 추가
    
    return merged

# ✂️ 5) refine_chunks_with_length_control 함수 - 길이 제한 재분할
# 💡 overlap(겹침)이 왜 필요? 청크 경계에서 문장이 잘리면 문맥이 끊겨요. 250자씩 겹치게 하면 앞 청크 끝부분이 다음 청크 앞부분에 다시 들어가서, 경계에 걸친 정보도 검색 가능해집니다.
# [청크1: ...... 마지막 250자]
#               [겹침]  [청크2: 처음 250자 ......]
def refine_chunks_with_length_control(
    chunks: List[dict],
    max_length: int = 1000,  # 최대 청크 길이
    overlap: int = 250       # 청크 간 겹침 길이
) -> List[dict]:
    """
    각 청크의 길이를 제한하면서 겹치는 영역을 포함해 추가 분할합니다.
    """
    refined = []  # 결과 리스트
    
    # LangChain의 재귀 분할기 생성
    # chunk_size: 한 조각 최대 크기 / chunk_overlap: 조각끼리 겹칠 크기
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, chunk_overlap=overlap
    )

    for chunk in chunks:
        # 병합된 섹션이 max_length보다 길면 더 작게 쪼갬
        split_texts = splitter.split_text(chunk["content"])
        
        # 쪼개진 각 조각을 순회 (i = 조각 순번)
        for i, split_text in enumerate(split_texts):
            refined.append(
                {
                    "title": chunk["title"],   # 원래 섹션 제목 유지
                    "content": split_text,      # 쪼개진 본문
                    "sub_chunk_idx": i,         # 같은 섹션 내 몇 번째 조각인지
                }
            )
    return refined

# 🏭 6) data_chunking 함수 - 전체 청킹 총괄 (메인 함수)
# 💡 이 함수가 곧 loader_main.py가 호출하던 그 함수예요. 3가지 분할 방식(section/recursive/token)을 지원하고, 기본값은 목차 기반인 section입니다.
def data_chunking(
    df: pd.DataFrame,
    splitter_type: str = "section",  # 분할 방식
    size: int = 1000,                # 청크 크기
    overlap: int = 250,              # 겹침
) -> List[Document]:
    """
    데이터프레임의 각 row를 청크 단위로 분할하고, langchain Document로 변환합니다.
    """
    # ===== 분할 방식에 따라 splitter 선택 =====
    if splitter_type == "recursive":
        # 문단/문장 단위로 자연스럽게 자르는 방식
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        )
    elif splitter_type == "token":
        # 토큰 개수 기준으로 자르는 방식
        splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    elif splitter_type == "section":
        # 목차 기반 방식 → 위에서 만든 함수들을 직접 쓸 거라 splitter 불필요
        splitter = None
    else:
        # 지원하지 않는 방식이면 에러
        raise ValueError(
            f"❌ [Value] (splitter.data_chunking.splitter_type) 지원하지 않는 분할 방식: {splitter_type}"
        )

    all_chunks = []  # 모든 파일의 청크를 모을 최종 리스트
    
    # DataFrame의 각 행(= 각 문서)을 순회
    for _, row in df.iterrows():
        # full_text 컬럼에서 본문 텍스트 가져옴 (없으면 빈 문자열)
        text = row.get("full_text", "")
        
        # 텍스트가 문자열이고 비어있지 않을 때만 처리
        if isinstance(text, str) and text.strip():
            try:
                # ===== section 방식 (목차 기반) =====
                if splitter_type == "section":
                    # 1) 목차 단위로 섹션 분리
                    sections = extract_sections(text)
                    if not sections:
                        print(f"⚠️ [Skip] 섹션 추출 실패로 청크 없음: {row.get('파일명')}")    

                    # 2) 각 섹션 본문 정제
                    cleaned_sections = []
                    for section in sections:
                        cleaned_section = {
                            "title": section["title"],
                            "content": clean_text(section["content"]),  # 텍스트 청소
                        }
                        cleaned_sections.append(cleaned_section)

                    # 3) 짧은 섹션 병합
                    merged = merge_short_chunks(cleaned_sections)
                    # 4) 긴 섹션은 길이 제한으로 재분할
                    chunks = refine_chunks_with_length_control(
                        merged, max_length=size, overlap=overlap
                    )
                    if not chunks:
                        print(f"⚠️ [Skip] 청크 0개 생성됨: {row.get('파일명')}")
                
                # ===== recursive / token 방식 =====
                else:
                    text = clean_text(text)             # 먼저 정제
                    chunks = splitter.split_text(text)  # 분할기로 쪼갬 (문자열 리스트 반환)

                # ===== 청크를 LangChain Document로 변환 =====
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        # section 방식이면 chunk가 dict라 ["content"] 사용
                        # recursive/token 방식이면 chunk가 문자열이라 그대로 사용
                        page_content=chunk["content"]
                        if isinstance(chunk, dict)
                        else chunk,
                        
                        # 메타데이터: 나중에 검색 결과에 출처를 표시하는 데 사용
                        metadata={
                            "사업명": row.get("사업명", ""),
                            "발주 기관": row.get("발주 기관", ""),
                            "파일명": row.get("파일명", ""),
                            "chunk_idx": i,  # 이 문서 내 청크 순번
                            # section 방식이면 섹션 제목도 메타데이터에 저장
                            "chunk_title": chunk.get("title", "")
                            if isinstance(chunk, dict)
                            else "",
                        },
                    )
                    all_chunks.append(doc)
            except Exception as e:
                # 한 파일 청킹 실패 시 전체 중단
                raise RuntimeError(
                    f"❌ [Runtime] (splitter.data_chunking) 청크 생성 오류 ({row.get('파일명')}): {e}"
                )
        else:
            # full_text가 비어있으면 건너뛰기
            print(f"⚠️ [Skip] full_text 비어있어 청크 건너뜸: {row.get('파일명')}")
            continue

    return all_chunks  # 모든 Document 청크 리스트 반환

# 🔍 7) inspect_sample_chunks 함수 - 샘플 청크 들여다보기
# 💡 디버깅용 함수예요. "청크가 제대로 잘렸나?"를 사람이 눈으로 확인할 때 씁니다.
def inspect_sample_chunks(
    chunks: List[Document], file_name: str, verbose: bool = False
) -> None:
    """
    특정 파일의 주요 청크(첫/중간/마지막/최장/최단)를 출력합니다.
    """
    # verbose가 꺼져있으면 아무것도 안 하고 종료
    if not verbose:
        return

    # 지정한 파일명에 해당하는 청크만 골라냄
    file_chunks = [doc for doc in chunks if doc.metadata.get("파일명") == file_name]
    if not file_chunks:  # 해당 파일 청크가 없으면 경고 후 종료
        print(f"❌ [Data] (splitter.inspect_sample_chunks) 청크 없음: {file_name}")
        return

    # 각 청크의 길이 리스트
    lengths = [len(doc.page_content) for doc in file_chunks]
    idx_max = lengths.index(max(lengths))  # 가장 긴 청크의 위치
    idx_min = lengths.index(min(lengths))  # 가장 짧은 청크의 위치

    # 살펴볼 대표 청크 5종 선정
    selected = {
        "첫 청크": file_chunks[0],
        "중간 청크": file_chunks[len(file_chunks) // 2],  # 정중앙 (정수 나눗셈)
        "마지막 청크": file_chunks[-1],                    # 마지막 (음수 인덱스)
        "가장 긴 청크": file_chunks[idx_max],
        "가장 짧은 청크": file_chunks[idx_min],
    }

    # 각 대표 청크의 길이와 내용 미리보기 출력
    for label, doc in selected.items():
        print(f"\n▶ {label}")
        print(f"  - 길이: {len(doc.page_content)}")
        print("  - 내용:")
        preview = doc.page_content[:300]  # 앞 300자만
        if len(doc.page_content) > 300:
            preview += "..."  # 잘렸으면 말줄임표
        print(f"    {preview}")
        
# 📊 8) summarize_chunk_quality 함수 - 청크 품질 통계
# 💡 500자 미만 비율이 핵심 지표예요. 이 비율이 높으면 청크가 너무 잘게 쪼개졌다는 뜻 = 검색 품질이 나쁠 수 있음. 그래서 이 비율이 높은 파일을 맨 위에 보여줘서 문제를 빨리 찾게 합니다.
def summarize_chunk_quality(
    chunks: List[Document], verbose: bool = False
) -> None:
    """
    파일별로 청크 수, 평균/최소/최대 길이, 500자 미만 비율 등을 요약 출력합니다.
    """
    if not verbose:  # verbose 꺼져있으면 종료
        return

    # 파일명 → 길이 리스트 형태로 모음
    # defaultdict(list): 없는 키 접근 시 자동으로 빈 리스트 생성
    summary = defaultdict(list)
    for doc in chunks:
        file_name = doc.metadata.get("파일명", "Unknown")
        length = len(doc.page_content)
        summary[file_name].append(length)  # 해당 파일의 길이 리스트에 추가

    # 파일별 통계 계산
    results = []
    for fname, lengths in summary.items():
        arr = np.array(lengths)  # NumPy 배열로 변환 (통계 계산 편리)
        results.append(
            {
                "파일명": fname,
                "청크수": len(arr),
                "평균길이": np.mean(arr),    # 평균
                "최소길이": np.min(arr),     # 최솟값
                "최대길이": np.max(arr),     # 최댓값
                # 500자 미만 청크의 비율 (%) = (500 미만 개수 / 전체) × 100
                "500자미만비율": np.sum(arr < 500) / len(arr) * 100,
            }
        )

    # 500자 미만 비율이 높은 순(품질 나쁜 순)으로 정렬
    # key=람다함수 / reverse=True → 내림차순
    results.sort(key=lambda x: x["500자미만비율"], reverse=True)

    # 결과 출력
    print("\n📌 청크 품질 요약")
    for res in results:
        print("=" * 60)  # 구분선
        print(f"📄 파일명: {res['파일명']}")
        print(f"  - 청크 수         : {res['청크수']}")
        print(f"  - 평균 길이       : {res['평균길이']:.2f}")  # 소수점 2자리
        print(f"  - 최소 길이       : {res['최소길이']}")
        print(f"  - 최대 길이       : {res['최대길이']}")
        print(f"  - 500자 미만 비율 : {res['500자미만비율']:.2f}%")
        # 각 파일의 샘플 청크도 함께 출력
        inspect_sample_chunks(chunks, res['파일명'], verbose=True)
        
        