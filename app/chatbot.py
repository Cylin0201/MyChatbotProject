from app.retriever import retriever, load_disease_titles, is_disease_in_query
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

# 모델 및 토크나이저 로드
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.half()
model = model.eval()

# 질환명 리스트 로드
disease_titles = load_disease_titles("data/emergency.jsonl")

def build_prompt(user_input: str, retrieved_docs: List[dict]) -> List[dict]:
    system_prompt = (
        "- AI 언어모델의 이름은 \"CLOVA X\" 이며 네이버에서 만들었다.\n"
        "- 오늘은 2025년 04월 29일(화)이다.\n"
        "- 인사말 및 기본적인 질문에 대해서는 문서를 참고하지 말고 일반적인 대화만 응답하라.\n"
        "- 사용자가 질문한 내용과 관련 있는 참고 문서를 아래에 제공하니, 가능한 그 정보를 반영해 답변하라.\n"
    )

    context_text = "\n\n".join([f"제목: {doc['title']}\n내용: {doc['content']}" for doc in retrieved_docs])

    user_prompt = (
        f"질문: {user_input}\n\n"
        f"[참고 문서]\n{context_text}\n\n"
        "위의 참고 문서를 근거로 답변해 주세요."
    )

    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return chat_history

def get_chatbot_response(user_input: str) -> dict:
    used_reference = is_disease_in_query(user_input, disease_titles)

    if used_reference:
        relevant_docs = retriever.retrieve(user_input, top_k=3)
        chat_history = build_prompt(user_input, relevant_docs)
    else:
        chat_history = [
            {"role": "system", "content": "사용자가 일반적인 대화를 요청했습니다. 문서를 참조하지 마세요."},
            {"role": "user", "content": user_input}
        ]

    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=False,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    assistant_prefix = "assistant\n"
    if assistant_prefix in decoded:
        response_text = decoded.split(assistant_prefix)[-1].strip()
    else:
        response_text = decoded.strip()

    for stop_token in ["<|endofturn|>", "<|stop|>", "<|im_end|>"]:
        if stop_token in response_text:
            response_text = response_text.split(stop_token)[0].strip()

    return {
        "response": response_text,
        "reference_used": used_reference
    }
