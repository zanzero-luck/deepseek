

--------------------------------------对话，推理------------------------------------------

直接调用deepseek_multichat就可以启动模型。

# 激活环境
cd ~
conda activate torch_env


# 推理，有以下方式
1. 终端对话
python deepseek_multichat.py
可以在代码中修改变量ckpt_path来选择模型

2. vllm加速，可以在代码中修改变量model来选择模型
python vllm_model.py



## 官方方式，https://github.com/deepseek-ai/DeepSeek-V3


--------------------------------------vllm-api部署与访问------------------------------------------
# 激活环境
cd ~
conda activate torch_env

#1.5B
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/model/DeepSeek-R1 \
  --served-model-name DeepSeek-R1 \
  --max-model-len=19000

#8B
python -m vllm.entrypoints.openai.api_server \
  --model /home/zhang/deepseek/DeepSeek-R1-Distill-Qwen-8B \
  --served-model-name DeepSeek-R1-Distill-Qwen-8B \
  --max-model-len=19000


  
终端运行以下指令：
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-R1-Distill-Qwen-8B",
        "messages": [
            {"role": "user", "content": "你是什么？<think>\n"}
        ]
    }'