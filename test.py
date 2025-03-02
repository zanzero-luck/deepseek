from model import OpenAIModel, print_with_color

configs = {
    "DEEPSEEK_API_BASE": "http://localhost:8000/v1/chat/completions",
    "DEEPSEEK_API_MODEL": "DeepSeek-R1",
    "MAX_TOKENS": 10240,
    "TEMPERATURE": 0,
    "OPENAI_API_KEY": ''
}

def ask(question: str):
    print_with_color("####################deepseek####################", "magenta")
    print_with_color(f"question: {question}", 'yellow')
    mllm = OpenAIModel(base_url=configs["DEEPSEEK_API_BASE"],
                    api_key=configs["OPENAI_API_KEY"],
                    model=configs["DEEPSEEK_API_MODEL"],
                    temperature=configs["TEMPERATURE"],
                    max_tokens=configs["MAX_TOKENS"],
                    disable_proxies=True)
    prompt = question
    status, rsp = mllm.get_model_response(prompt)
    if not status:
        print_with_color(f"失败，{rsp}", 'red')
        return
    print_with_color(f"*********************** rsp:\n{rsp}", "yellow")


ask("请计算5的阶乘")