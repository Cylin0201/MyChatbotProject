from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 및 토크나이저 로드
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.half()
model = model.eval()

chat_history = [
    {"role": "system", "content": "- AI 언어모델의 이름은 \"CLOVA X\" 이며 네이버에서 만들었다.\n"
    "- 오늘은 2025년 04월 27일(일)이다.\n"
    "- 인사말 및 기본적인 질문에 대해서는 문서를 참고하지 말고 일반적인 대화만 응답하라. "}
]

def get_chatbot_response(user_input: str) -> str:
    chat_history.append({"role": "user", "content": user_input})

    # 입력 텐서 생성
    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=False,  # padding 안함
        truncation=True # 너무 긴 이력 자름
    ).to(device)

    # 모델 추론
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_k=50,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 디코딩
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 'assistant' 이후 텍스트만 추출
    assistant_prefix = "assistant\n"
    if assistant_prefix in decoded:
        response = decoded.split(assistant_prefix)[-1].strip()
    else:
        response = decoded.strip()

    # <|endofturn|>, <|stop|>, <|im_end|> 모두 제거
    for stop_token in ["<|endofturn|>", "<|stop|>", "<|im_end|>"]:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()

    return response
