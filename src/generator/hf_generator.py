# 소스코드 분석 (2026.05.20 minjae)
# 참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3

# 코드를 한 줄씩 분석해서 주석을 달아드릴게요. 이 파일은 Hugging Face 모델을 불러와서(load) 답변을 생성(generate)하는 두 가지 함수로 구성되어 있어요.

# 🎯 핵심 흐름 요약
# 전체 동작을 한눈에 보면 이렇게 흘러갑니다:

# 모델 준비 단계 (load_hf_model)
# config에서 모델 이름을 읽고 → 토크나이저 로딩 → (옵션) 4bit 양자화 적용 → GPU에 모델 배치 → 추론 모드 전환
# 답변 생성 단계 (generate_answer_hf)
# 프롬프트를 토큰으로 변환 → GPU에서 generate() 실행 → 입력 부분 제거 → stop word 자르기 → 불필요 어구 제거 → 너무 짧으면 폴백 메시지 → 반환
# 결정론적 생성
# do_sample=False, num_beams=1로 설정했기 때문에 같은 프롬프트에는 항상 같은 답변이 나옵니다. RAG 시스템에서 일관성을 위해 흔히 쓰는 설정이에요.

# 추가로 궁금한 부분(예: repetition_penalty가 어떻게 작동하는지, device_map="auto"가 정확히 뭘 하는지 등)이 있으면 더 깊이 설명해드릴 수 있어요!

# 📦 1) Import 부분 (외부 라이브러리 불러오기)
import os  # 운영체제 관련 모듈 (환경변수 읽기, 파일 경로 처리 등에 사용)
import torch  # PyTorch 딥러닝 프레임워크 (GPU 연산, 텐서 처리)
from typing import Dict  # 타입 힌트용 - 함수 인자/반환값이 '딕셔너리'임을 명시
from inspect import signature  # 함수의 매개변수 목록을 런타임에 확인할 때 사용
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# AutoTokenizer: 모델 이름만 주면 그에 맞는 토크나이저(텍스트→숫자 변환기)를 자동 로딩
# AutoModelForCausalLM: 인과적 언어모델(다음 단어 예측 GPT 계열)을 자동 로딩
# BitsAndBytesConfig: 모델을 4bit/8bit로 양자화(경량화)할 때 쓰는 설정 클래스
from langsmith import trace  # LangSmith 트레이싱 - 모델 호출을 로깅/모니터링하는 도구

# 💡 양자화(Quantization)란? 모델의 가중치를 32bit → 4bit처럼 더 적은 비트로 압축해서 GPU 메모리를 적게 쓰도록 만드는 기술이에요. 7B 모델도 16GB GPU에 올릴 수 있게 해줍니다.

# 🔧 2) load_hf_model 함수 - 모델 로딩
def load_hf_model(config: Dict) -> Dict:
    """
    Hugging Face 모델을 config를 기반으로 초기화합니다.
    Args:
        config (Dict): 설정 정보 (모델 이름, 양자화 여부 등 포함)
    Returns:
        Dict: 모델 정보 및 토크나이저
    """
    # config.yaml에서 "generator" 섹션의 "model_name" 값을 꺼냄
    # 예: "Markr-AI/Gukbap-Qwen2.5-7B" 같은 허깅페이스 모델 ID
    model_name = config["generator"]["model_name"]

    # 토크나이저 로딩: 사람이 읽는 텍스트를 모델이 이해하는 숫자(token id)로 변환하는 도구
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,        # Rust 기반 빠른 토크나이저 대신 Python 기반 사용 (호환성 ↑)
        trust_remote_code=True, # 모델 저장소의 커스텀 코드 실행 허용 (일부 모델에 필수)
        token=os.getenv("HF_TOKEN")  # 환경변수에서 HuggingFace 인증 토큰 읽기 (비공개 모델 접근용)
    )

    # config에서 양자화 사용 여부를 가져옴 (없으면 기본값 False)
    use_quantization = config["generator"].get("use_quantization", False)

    if use_quantization:
        # 양자화는 GPU에서만 동작하므로, GPU가 없으면 에러 발생시킴
        if not torch.cuda.is_available():
            raise EnvironmentError("❌ GPU가 사용 가능하지 않습니다. 양자화 모델을 사용하려면 GPU가 필요합니다.")
        
        # 양자화 세부 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                      # 모델 가중치를 4bit로 압축해서 로딩
            bnb_4bit_use_double_quant=True,         # 이중 양자화 (메모리를 더 절약하는 기법)
            bnb_4bit_quant_type="nf4",              # NF4(NormalFloat4) 방식 - 정확도가 가장 좋은 4bit 양자화
            bnb_4bit_compute_dtype=torch.float16,   # 연산 시에는 float16으로 변환 (속도 ↑)
            llm_int8_enable_fp32_cpu_offload=True,  # GPU 부족 시 일부 레이어를 CPU(fp32)로 옮김
        )
        # 양자화 설정을 적용한 모델 로딩
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,    # 위에서 만든 양자화 설정 적용
            trust_remote_code=True,            # 모델의 커스텀 코드 실행 허용
            token=os.getenv("HF_TOKEN"),       # 인증 토큰
            device_map="auto"                  # 모델 레이어를 GPU/CPU에 자동 분배 배치
        )
    else:
        # 양자화 없이 원본 정밀도로 모델 로딩 (메모리는 많이 쓰지만 정확도 최대)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            device_map="auto"
        )

    # 모델을 추론 모드로 전환 (Dropout/BatchNorm 비활성화 - 학습이 아닌 답변 생성용이므로)
    model.eval()
    
    # 토크나이저와 모델을 딕셔너리로 묶어서 반환 → 이후 generate 함수에서 사용
    return {"tokenizer": tokenizer, "model": model}

# 💬 3) generate_answer_hf 함수 - 답변 생성
def generate_answer_hf(prompt: str, model_info: Dict, generation_config: Dict) -> str:
    """
    Hugging Face 모델을 사용하여 프롬프트에 응답을 생성합니다
    """
    # LangSmith 트레이싱 시작: 이 블록 안에서 일어나는 모든 일을 기록/추적
    # 디버깅 및 성능 모니터링에 유용
    with trace(name="generate_answer_hf", inputs={"prompt": prompt}) as run:
        # 앞서 load_hf_model이 반환한 딕셔너리에서 꺼냄
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        # 프롬프트 텍스트를 모델이 이해하는 텐서(숫자 배열)로 변환
        # return_tensors="pt"는 PyTorch 텐서 형식으로 반환하라는 의미
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # input_ids: 토큰 ID 시퀀스. 모델이 위치한 디바이스(GPU/CPU)로 이동
        input_ids = inputs["input_ids"].to(model.device)
        
        # attention_mask: 실제 단어(1)와 패딩(0)을 구분하는 마스크. 마찬가지로 디바이스 이동
        attention_mask = inputs["attention_mask"].to(model.device)

        # 모델.generate()에 넘길 인자들을 딕셔너리로 정리
        generate_kwargs = {
            "input_ids": input_ids,           # 입력 토큰
            "attention_mask": attention_mask, # 어텐션 마스크
            "max_new_tokens": generation_config.get("max_length", 512),  # 최대로 생성할 새 토큰 수 (기본 512)
            "do_sample": False,               # 확률적 샘플링 끔 → 결정론적(똑같은 입력에 똑같은 출력)
            "num_beams": 1,                   # 빔서치 끔 (1이면 그리디 디코딩 - 가장 확률 높은 단어만 선택)
            "temperature": None,              # 샘플링 온도 (do_sample=False라 사용 안 함)
            "top_k": None,                    # Top-K 샘플링 (마찬가지로 비활성화)
            "top_p": None,                    # Top-P(누클리어스) 샘플링 (마찬가지로 비활성화)
            "eos_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
            # 문장 종료 토큰 ID 지정 (이 토큰이 나오면 생성 중지) - eos가 없으면 pad로 대체
            "repetition_penalty": 1.2         # 반복 페널티 - 같은 단어가 반복되면 확률을 깎음 (1.0=꺼짐)
        }

        # token_type_ids는 BERT 계열에선 쓰지만 GPT/Qwen 같은 모델은 안 씀
        # 입력에 있고, 모델 generate가 받아주는 경우에만 추가 (호환성 처리)
        if "token_type_ids" in inputs and "token_type_ids" in signature(model.generate).parameters:
            generate_kwargs["token_type_ids"] = inputs["token_type_ids"].to(model.device)

        # 그래디언트 계산 비활성화 (메모리 절약 + 속도 향상, 추론 시에만 사용)
        with torch.no_grad():
            output = model.generate(**generate_kwargs)  # ** 로 딕셔너리를 키워드 인자로 풀어서 전달

        # ===== 여기부터 후처리 =====
        
        # output은 배치 형태라 첫 번째 결과만 추출 (배치 크기 1이므로)
        output_ids = output[0]
        
        # 입력 프롬프트의 토큰 길이를 구함
        input_len = input_ids.size(1)
        
        # 출력 중에서 "입력 부분을 제외한 새로 생성된 토큰들"만 잘라냄
        # → 답변에서 원본 프롬프트가 함께 출력되는 것을 방지
        generated_ids = output_ids[input_len:]
        
        # 토큰 ID를 다시 사람이 읽는 텍스트로 변환
        # skip_special_tokens=True: <eos>, <pad> 같은 특수 토큰 제거
        # clean_up_tokenization_spaces=True: 토크나이저가 만든 불필요한 공백 정리
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # 앞뒤 공백/줄바꿈 제거
        answer = generated_text.strip()

        # 모델이 가끔 코드블록(```)이나 "Human:" 같은 토큰을 생성하면서 헛소리를 이어가는 경우가 있음
        # 이런 stop word가 나오면 그 앞부분만 잘라서 사용
        stop_strings = ["```", "<|endoftext|>", "Human:", "human:", "###"]
        for stop_str in stop_strings:
            if stop_str in answer:
                answer = answer.split(stop_str)[0].strip()  # 앞부분만 남김

        # 모델이 자주 반복하는 무의미한 어구를 후처리로 제거
        # (한국어 LLM에서 자주 나오는 빈말/어미들)
        bad_tokens = ["하십시오", "하실 수", "알고 싶어요", "하는데 필요한", "것을", "한다", "하십시오.", "하시기 바랍니다"]
        for token in bad_tokens:
            answer = answer.replace(token, "")  # 단순 치환으로 제거

        # 답변이 너무 짧거나(10자 미만), 공백이 너무 적은 경우(=의미 없는 출력)
        # → 사용자에게 명확한 메시지를 보여주기 위해 폴백 응답으로 교체
        if len(answer) < 10 or answer.count(" ") < 3:
            answer = "해당 문서에서 질문에 대한 명확한 정보를 찾을 수 없습니다"

        # LangSmith에 최종 출력 기록 (모니터링용)
        run.add_outputs({"output": answer})
        
        # 정제된 최종 답변 반환
        return answer