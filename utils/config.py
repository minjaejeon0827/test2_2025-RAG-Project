# 소스코드 분석 (2026.05.28 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 이 파일은 src/utils/에 있는 설정 관리 모듈이에요. 지금까지 본 모든 파일이 config["data"]["top_k"] 같은 설정값을 가져다 썼는데, 그 설정값들이 올바른지 미리 검사하고 불러오는 역할을 합니다. 일종의 "문지기" 같은 거죠.

# 🔄 전체 흐름 요약
# chatbot.py / main.py
#    │
#    ▼
# load_config(project_root)        ← config.yaml 읽기
#    │
#    ├─ yaml.safe_load()            → 파일을 딕셔너리로 변환
#    ├─ project_root 주입
#    └─ check_config(config)        → 모든 설정값 검증
#          ├─ settings   (verbose, project_root)
#          ├─ data       (top_k, file_type, splitter, chunk_size...)
#          ├─ embedding  (embed_model, db_type, vector_db_path)
#          ├─ retriever  (search_type, top_k, rerank...)
#          ├─ generator  (model_type, model_name, max_length...)
#          └─ chat_history (list 검사)
#    │
#    ▼
# return config                    ← 검증된 설정을 전체 시스템에 공급

# 🎯 이 파일의 역할 한 줄 요약

# "config.yaml을 읽어서 → 모든 설정값이 올바른 타입/범위인지 검사하고 → 안전한 설정 딕셔너리를 시스템 전체에 공급하는 문지기"

# 지금까지 본 모든 파일(splitter.py, retrieval.py, hf_generator.py 등)이 사용하던 config["data"]["splitter"], config["retriever"]["search_type"] 같은 값들이 여기서 미리 검증되는 거예요. 그래서 뒤쪽 코드들은 "설정값은 이미 올바르다"고 믿고 안심하고 쓸 수 있죠.
# 💡 이 파일의 설계가 좋은 이유
# **방어적 프로그래밍(defensive programming)**의 좋은 예시예요. 사용자가 config.yaml에 db_type: postgresql 같은 지원 안 되는 값을 적으면, 프로그램 시작 단계에서 바로 잡아냅니다. 이게 없으면 한참 실행되다가 벡터DB 생성 단계에서 알 수 없는 에러가 터지는데, 그때 디버깅하기는 훨씬 어렵거든요. "에러는 최대한 빨리, 명확하게"라는 원칙을 잘 지킨 코드입니다.
# 궁금한 부분(예: config.get(키, 기본값) 패턴, isinstance() 동작, yaml.safe_load와 yaml.load의 차이, try/except의 예외 순서 등) 있으면 더 자세히 설명해드릴게요!

# 📦 1) Import 부분
import os    # 파일 경로 처리, 경로 존재 여부 확인
import yaml  # YAML 파일(config.yaml) 읽기/파싱

# ✅ 2) check_config 함수 - 설정 유효성 검사
# 이 함수는 길지만 패턴이 똑같아요. "각 설정값을 꺼내서 → 타입/값이 올바른지 확인 → 틀리면 에러" 의 반복입니다.
def check_config(config: dict) -> None:
    """
    설정 딕셔너리의 유효성을 검사합니다.
    """
    # config 자체가 딕셔너리가 아니면 에러 (가장 기본 검사)
    if not isinstance(config, dict):
        raise ValueError("❌ [Value] (config.check_config) 설정은 딕셔너리여야 합니다.")
    
    # 🔧 settings 섹션 검사
    # 💡 config.get("settings", {})의 {}는 기본값이에요. "settings" 키가 없으면 빈 딕셔너리를 대신 쓴다는 뜻. KeyError 방지용입니다.
    # config에서 "settings" 부분을 꺼냄 (없으면 빈 딕셔너리)
    settings_config = config.get("settings", {})
    if not isinstance(settings_config, dict):
        raise ValueError("❌ (config.check_config.settings) 설정은 딕셔너리여야 합니다.")
    else:
        # verbose: 상세 로그 출력 여부 → 반드시 True/False(bool)여야 함
        verbose = settings_config.get("verbose", False)
        if not isinstance(verbose, bool):
            raise ValueError("❌ (config.check_config.settings.verbose) verbose는 True 또는 False여야 합니다.")
        
        # project_root: 프로젝트 루트 경로 → 문자열이어야 함
        project_root = settings_config.get("project_root", "2025-LLM-Project")
        if not isinstance(project_root, str):
            raise ValueError("❌ (config.check_config.settings.project_root) project_root는 문자열이어야 합니다.")
        
    # 📂 data 섹션 검사
    # ⚠️ 작은 주의점: chunk_size < 1 or not isinstance(chunk_size, int)에서 만약 chunk_size가 문자열이면 < 1 비교에서 먼저 에러가 날 수 있어요. 순서상 not isinstance(...)를 앞에 두는 게 더 안전합니다. (지금 코드도 대부분 정수가 들어와서 큰 문제는 없어요.)
    # 데이터 관련 설정 꺼내기
    data_config = config.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("❌ (config.check_config.data) 데이터 설정은 딕셔너리여야 합니다.")
    else:
        # folder_path: 원본 파일 폴더 경로
        folder_path = data_config.get("folder_path", "data/files")
        if not isinstance(folder_path, str):  # 문자열 검사
            raise ValueError("❌ (config.check_config.data.folder_path) 폴더 경로는 문자열이어야 합니다.")
        if not os.path.exists(folder_path):    # 실제 존재 검사
            raise FileNotFoundError(f"❌ (config.check_config.data.folder_path) 폴더 경로가 존재하지 않습니다: {folder_path}")
        
        # data_list_path: 메타데이터 CSV 경로 (앞서 data_loader.py에서 쓰던 그 CSV)
        data_list_path = data_config.get("data_list_path", "data/data_list.csv")
        if not isinstance(data_list_path, str):
            raise ValueError("❌ (config.check_config.data.data_list_path) 데이터 목록 경로는 문자열이어야 합니다.")
        if not os.path.exists(data_list_path):
            raise FileNotFoundError(f"❌ (config.check_config.data.data_list_path) 파일 목록 경로가 존재하지 않습니다: {data_list_path}")
        
        # top_k: 검색할 문서 수 → 정수, 1~100 범위 권장
        top_k = data_config.get("top_k", 5)
        if not isinstance(top_k, int):
            raise ValueError("❌ (config.check_config.data.top_k) top_k는 정수여야 합니다.")
        if top_k < 1:
            print("⚠️ (config.check_config.data.top_k) top_k는 1이상의 정수여야 합니다.")  # 경고만 (에러 아님)
        if top_k > 100:
            print("⚠️ (config.check_config.data.top_k) top_k는 100이하의 정수여야 합니다.")

        # file_type: 처리할 파일 종류 → 정해진 3개 중 하나만 허용
        file_type = data_config.get("file_type", "all")
        if file_type not in ["hwp", "pdf", "all"]:
            raise ValueError("❌ (config.check_config.data.file_type) file_type은 'hwp', 'pdf', 'all' 중 하나여야 합니다.")
        
        # apply_ocr: OCR 적용 여부 → bool
        apply_ocr = data_config.get("apply_ocr", False)
        if not isinstance(apply_ocr, bool):
            raise ValueError("❌ (config.check_config.data.apply_ocr) apply_ocr는 True 또는 False여야 합니다.")
        
        # splitter: 분할 방식 → 앞서 splitter.py에서 본 3가지 중 하나
        splitter = data_config.get("splitter", "section")
        if not isinstance(splitter, str):
            raise ValueError("❌ (config.check_config.data.splitter) 청크 분할기 타입은 문자열이어야 합니다.")
        if splitter not in ["recursive", "token", "section"]:
            raise ValueError("❌ (config.check_config.data.splitter) splitter는 'recursive', 'token', 'section' 중 하나여야 합니다.")
        
        # chunk_size: 청크 크기 → 1 이상 정수
        chunk_size = data_config.get("chunk_size", 1000)
        if chunk_size < 1 or not isinstance(chunk_size, int):
            raise ValueError("❌ (config.check_config.data.chunk_size) 청크 크기는 1 이상의 정수여야 합니다.")
        
        # chunk_overlap: 청크 겹침 → 0 이상 정수
        chunk_overlap = data_config.get("chunk_overlap", 250)
        if chunk_overlap < 0 or not isinstance(chunk_overlap, int):
            raise ValueError("❌ (config.check_config.data.chunk_overlap) 청크 오버랩은 0 이상의 정수여야 합니다.")
        
    # 🧠 embedding 섹션 검사
    embedding_config = config.get("embedding", {})
    if not isinstance(embedding_config, dict):
        raise ValueError("❌ (config.check_config.embedding) 임베딩 설정은 딕셔너리여야 합니다.")
    else:
        # embed_model: 임베딩 모델 이름 (예: KoE5) → 문자열
        # (변수명이 embed_mode로 오타지만 동작엔 문제없음)
        embed_mode = embedding_config.get("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if not isinstance(embed_mode, str):
            raise ValueError("❌ (config.check_config.embedding.embed_model) 임베딩 모델 이름은 문자열이어야 합니다.")

        # db_type: 벡터DB 종류 → faiss 또는 chroma만 허용
        db_type = embedding_config.get("db_type", "faiss")
        if db_type not in ["faiss", "chroma"]:
            raise ValueError("❌ (config.check_config.embedding.db_type) 지원하지 않는 벡터 DB 타입입니다. ('faiss' 또는 'chroma' 사용)")
        
        # vector_db_path: 벡터DB 저장 경로 → 문자열 + 실제 존재
        vector_db_path = embedding_config.get("vector_db_path", "data/vector_db")
        if not isinstance(vector_db_path, str):
            raise ValueError("❌ (config.check_config.embedding.vector_db_path) 벡터 DB 경로는 문자열이어야 합니다.")
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"❌ (config.check_config.embedding.vector_db_path) 벡터 DB 경로가 존재하지 않습니다: {vector_db_path}")
        
    # 🔍 retriever 섹션 검사
    # 💡 data의 top_k(문서 수)와 retriever의 top_k(청크 수)는 이름은 같지만 다른 값이에요. 데이터는 "몇 개의 파일", 리트리버는 "몇 개의 청크"를 뜻합니다.
    retriever_config = config.get("retriever", {})
    if not isinstance(retriever_config, dict):
        raise ValueError("❌ (config.check_config.retriever) 검색기 설정은 딕셔너리여야 합니다.")
    else:
        # query: 사용자 질문 → 문자열 (비어도 됨)
        query = retriever_config.get("query", "")
        if not isinstance(query, str):
            raise ValueError("❌ (config.check_config.retriever.query) 쿼리는 문자열이어야 합니다.")
        
        # search_type: 검색 방식 → similarity 또는 hybrid (retrieval.py에서 본 그것!)
        search_type = retriever_config.get("search_type", "hybrid")
        if search_type not in ["similarity", "hybrid"]:
            raise ValueError("❌ (config.check_config.retriever.search_type) search_type은 'similarity' 또는 'hybrid' 중 하나여야 합니다.")

        # top_k: 검색 청크 수 → 정수
        top_k = retriever_config.get("top_k", 10)
        if not isinstance(top_k, int):
            raise ValueError("❌ (config.check_config.retriever.top_k) top_k는 정수여야 합니다.")
        if top_k < 1:
            print("⚠️ (config.check_config.retriever.top_k) top_k는 1이상의 정수여야 합니다.")
        
        # rerank: 재정렬 사용 여부 → bool (CrossEncoder 사용 여부)
        rerank = retriever_config.get("rerank", True)
        if not isinstance(rerank, bool):
            raise ValueError("❌ (config.check_config.retriever.rerank) rerank는 True 또는 False여야 합니다.")

        # rerank_top_k: 재정렬 후 최종 청크 수 → 정수
        rerank_top_k = retriever_config.get("rerank_top_k", 5)
        if not isinstance(rerank_top_k, int):
            raise ValueError("❌ (config.check_config.retriever.rerank_top_k) rerank_top_k는 정수여야 합니다.")
        if rerank_top_k < 1:
            print("⚠️ (config.check_config.retriever.rerank_top_k) rerank_top_k는 1 이상의 정수여야 합니다.")
            
    # 🤖 generator 섹션 검사
    generator_config = config.get("generator", {})
    if not isinstance(generator_config, dict):
        raise ValueError("❌ (config.check_config.generator) 생성기 설정은 딕셔너리여야 합니다.")
    else:
        # model_type: 생성 모델 종류 → huggingface 또는 openai
        model_type = generator_config.get("model_type", "huggingface")
        if model_type not in ["huggingface", "openai"]:
            raise ValueError("❌ (config.check_config.generator.model_type) 지원하지 않는 모델 타입입니다. ('huggingface' 또는 'openai' 사용)")
        
        # model_name: 모델 이름 → 문자열
        model_name = generator_config.get("model_name", "")
        if not isinstance(model_name, str):
            raise ValueError("❌ (config.check_config.generator.model_name) 모델 이름은 문자열이어야 합니다.")
        
        # max_length: 최대 생성 토큰 수 → 1 이상 정수
        max_length = generator_config.get("max_length", 512)
        if not isinstance(max_length, int):
            raise ValueError("❌ (config.check_config.generator.max_length) 최대 길이는 정수여야 합니다.")
        if max_length < 1:
            raise ValueError("❌ (config.check_config.generator.max_length) 최대 길이는 1이상의 정수여야 합니다.")
        
        # use_quantization: 양자화 사용 여부 → bool (hf_generator.py에서 본 그것!)
        use_quantization = generator_config.get("use_quantization", True)
        if not isinstance(use_quantization, bool):
            raise ValueError("❌ (config.check_config.generator.use_quantization) 양자화는 True/ False여야 합니다.")
    
    # ===== chat_history 검사 =====
    # 대화 이력 → 반드시 리스트여야 함
    chat_history = config.get("chat_history")
    if not isinstance(chat_history, list):
        raise ValueError("❌ (config.check_config.chat_history) chat_history는 list이어야 합니다.")
    
# 📂 3) load_config 함수 - 설정 파일 로딩
# ⚠️ 주목할 점: load_config는 check_config에서 에러가 나도 except로 잡아서 print만 하고 계속 진행해요. 즉, 검증에 실패해도 (잘못된) config를 그대로 반환합니다. 이건 의도적일 수도 있지만, 잘못된 설정으로 뒤에서 더 큰 에러가 날 수 있는 부분이에요.
def load_config(project_root: str) -> dict:
    """
    YAML 파일에서 설정을 로드합니다.
    """
    # 프로젝트 루트 + "config.yaml" → 설정 파일 경로 조합
    config_path = os.path.join(project_root, "config.yaml")
    
    # 파일이 없으면 에러
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ (config.load_config) 설정 파일을 찾을 수 없습니다: {config_path}")

    # YAML 파일 열어서 파이썬 딕셔너리로 변환
    # encoding='utf-8' → 한글 깨짐 방지
    # safe_load → 안전한 로딩 (악성 코드 실행 방지)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # settings 섹션이 없으면 빈 딕셔너리로 생성
    if "settings" not in config or not isinstance(config["settings"], dict):
        config["settings"] = {}
    # 전달받은 project_root를 settings에 강제로 넣음
    # → 코드 실행 위치에 따라 동적으로 경로 설정
    config["settings"]["project_root"] = project_root

    try:
        # 위에서 만든 유효성 검사 함수 실행
        check_config(config)

        # (주석 처리됨) verbose 모드일 때 전체 설정 출력하는 코드
        #if config.get("settings", {}).get("verbose", False):
            #print("\n📄 [Verbose] 최종 설정 내용:")
            #print(yaml.dump(config, allow_unicode=True, sort_keys=False))
    
    # ===== 예외별 처리 =====
    # 파일 접근 관련 오류
    except (FileNotFoundError, PermissionError) as e:
        print(f"❌ [File] 파일 접근 오류:\n  {e}")
    # YAML 문법 오류
    except yaml.YAMLError as e:
        print(f"❌ [YAML] 설정 파일 파싱 오류:\n  {e}")
    # 설정값 타입/값 오류
    except (ValueError, TypeError) as e:
        print(f"❌ [Config] 설정값 오류:\n  {e}")
    # 그 외 예상치 못한 모든 오류
    except Exception as e:
        print(f"❌ [Unexpected] 예상치 못한 오류 발생:\n  {e}")

    # 검증된 config 딕셔너리 반환
    return config