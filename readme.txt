
下载模型示例：
modelscope download --model unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf --loca
l_dir ./DeepSeek-R1-Distill-Qwen-14B-Q4
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

pip install vllm

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


--------------------------------------llama.cpp部署与访问------------------------------------------
wget -c https://github.com/ggml-org/llama.cpp/releases/download/b4798/llama-b4798-bin-ubuntu-x64.zip
cd /home/zhang/build/bin

#执行如下命令开启服务
./llama-server     --model /home/zhang/deepseek/DeepSeek-R1-Distill-Qwen-14B-Q4/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf     --port 10000     --ctx-size 1024     --n-gpu-
layers 40



curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Hello, how are you?", "max_tokens": 50}' http://127.0.0.1:10000/v1/chat/completions