import re
import json
from abc import abstractmethod
from typing import List
from http import HTTPStatus
import base64
import requests
import dashscope
import sys
from typing import Tuple
from colorama import Fore, Style

sys.path.append('./mswift')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def print_with_color(text: str, color=""):
    if color == "red":
        print(Fore.RED + text)
    elif color == "green":
        print(Fore.GREEN + text)
    elif color == "yellow":
        print(Fore.YELLOW + text)
    elif color == "blue":
        print(Fore.BLUE + text)
    elif color == "magenta":
        print(Fore.MAGENTA + text)
    elif color == "cyan":
        print(Fore.CYAN + text)
    elif color == "white":
        print(Fore.WHITE + text)
    elif color == "black":
        print(Fore.BLACK + text)
    else:
        print(text)
    print(Style.RESET_ALL)

class BaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str, images: List[str]) -> Tuple[bool, str]:
        pass


class OpenAIModel(BaseModel):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, max_tokens: int, disable_proxies=False):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_proxies = disable_proxies

    def get_model_response(self, prompt: str, images: List[str]=[], tools: list[dict]=None,
                        history: list[dict]=None) -> Tuple[bool, str]:
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            # "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if tools:
            payload["tools"] = tools
        if history:
            payload['messages'] = history.append(payload['messages'][-1])
        if self.disable_proxies:
            response = requests.post(self.base_url, headers=headers, json=payload, proxies={}).json()
        else:
            response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in str(response):
            if not 'usage' in response:
                print_with_color(f"not usage:{response}", 'res')
            else:
                usage = response["usage"]
                prompt_tokens = usage["prompt_tokens"]
                total_tokens = usage["total_tokens"]
                completion_tokens = usage["completion_tokens"]
                print_with_color(f"total_tokens: {total_tokens}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")
                completion_tokens = usage["completion_tokens"]
                if self.model == "gpt-4o":
                    print_with_color(f"Request gpt-4o cost is "
                                f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.005 + completion_tokens / 1000 * 0.015)}",
                                "yellow")
                else:
                    print_with_color(f"Request cost is "
                                    f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03)}",
                                    "yellow")
        else:
            print_with_color(f"执行失败，response: {response}", "red")
            return False, response
        if tools:
            return True, response["choices"][0]["message"]["tool_calls"]
        else:
            return True, response["choices"][0]["message"]["content"]

class SwiftModel(BaseModel):
    def __init__(self, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_model_response(self, prompt: str, images: List[str], stream=False) -> (bool, str):
        system = ""
        from mswift.swift.llm import XRequestConfig, inference_client
        if stream:
            request_config = XRequestConfig(stream=True, temperature=self.temperature, max_tokens=self.max_tokens)
            stream_resp = inference_client(self.model, prompt, system=system, images=images, request_config=request_config)
            print(f'query: {prompt}')
            response_content = ""  # 用于存储完整的响应内容
            print('response: ', end='')

            for chunk in stream_resp:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)  # 实时打印
                response_content += content  # 将当前chunk的内容添加到response_content

            return True, response_content  # 返回成功标志和完整的响应内容
            
        else:
            request_config = XRequestConfig(stream=False, temperature=self.temperature, max_tokens=self.max_tokens)
            resp = inference_client(self.model, prompt, images=images, request_config=request_config)
            print(f'query: {prompt}')
            print('response: ', resp)
            stream_resp = resp.choices[0].message.content
            return True, stream_resp  # 返回成功标志和完整的响应内容

class QwenModel(BaseModel):
    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.model = model
        dashscope.api_key = api_key

    def get_model_response(self, prompt: str, images: List[str]) -> Tuple[bool, str]:
        content = [{
            "text": prompt
        }]
        for img in images:
            img_path = f"file://{img}"
            content.append({
                "image": img_path
            })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        response = dashscope.MultiModalConversation.call(model=self.model, messages=messages)
        if response.status_code == HTTPStatus.OK:
            return True, response.output.choices[0].message.content[0]["text"]
        else:
            return False, response.message


def parse_explore_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            area = int(re.findall(r"tap\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "text":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "long_press":
            area = int(re.findall(r"long_press\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0]
            area, swipe_dir, dist = params.split(",")
            area = int(area)
            swipe_dir = swipe_dir.strip()[1:-1]
            dist = dist.strip()[1:-1]
            return [act_name, area, swipe_dir, dist, last_act]
        elif act_name == "grid":
            return [act_name]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR1: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]


def parse_grid_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            params = re.findall(r"tap\((.*?)\)", act)[0].split(",")
            # print("params",params)
            x = int(params[0].strip())
            # print("x",x)
            y = int(params[1].strip())
            # print("y",y)
            return [act_name, x, y, last_act]
        elif act_name == "long_press":
            params = re.findall(r"long_press\((.*?)\)", act)[0].split(",")
            x = int(params[0].strip())
            # print("x",x)
            y = int(params[1].strip())
            # print("y",y)
            return [act_name, x, y, last_act]
        elif act_name == "text":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0].split(",")
            if len(params) != 4:
                print_with_color(f"ERROR: swipe param: {params}!", "red")
                return ["ERROR"]
            x = int(params[0].strip())
            # print("x",x)
            y = int(params[1].strip())
            # print("y",y)
            direction = params[2].strip()
            distance = params[3].strip()
            return [act_name, x, y, direction, distance, last_act]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR2: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]


def parse_reflect_rsp(rsp):
    try:
        decision = re.findall(r"Decision: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Decision:", "yellow")
        print_with_color(decision, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        if decision == "INEFFECTIVE":
            return [decision, think]
        elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
            doc = re.findall(r"Documentation: (.*?)$", rsp, re.MULTILINE)[0]
            print_with_color("Documentation:", "yellow")
            print_with_color(doc, "magenta")
            return [decision, think, doc]
        else:
            print_with_color(f"ERROR: Undefined decision {decision}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR3: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]

def predict(prompt, images, model_name="OpenAI", max_tokens=-1):
    from .config import load_config
    configs = load_config()
    if max_tokens == -1:
        max_tokens=configs["MAX_TOKENS"]
    if model_name == "OpenAI":
        mllm = OpenAIModel(base_url=configs["OPENAI_API_BASE"],
                            api_key=configs["OPENAI_API_KEY"],
                            model=configs["OPENAI_API_MODEL"],
                            temperature=configs["TEMPERATURE"],
                            max_tokens=max_tokens)
    elif model_name == "Qwen":
        mllm = QwenModel(api_key=configs["DASHSCOPE_API_KEY"],
                            model=configs["QWEN_MODEL"])
    else:
        print_with_color(f"ERROR: Unsupported model type {model_name}!", "red")
    print_with_color(f"prompt:{prompt}", 'blue')
    status, rsp = mllm.get_model_response(prompt, images)
    if status:
        print_with_color(f"result:{rsp}", "blue")
    else:
        print_with_color(rsp, "red")
    
    return status, rsp

def predict2(question, config):
    api_key = config["OPENAI_API_KEY"]
    url = config["OPENAI_API_BASE"]
    model = config["OPENAI_T2T_MODEL"]
    max_tokens = config["MAX_TOKENS"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        'messages': [
            {'role': 'user', 'content': question}
        ],
        'max_tokens': max_tokens
    }

    response = requests.post(url, headers=headers, json=data)
    success = False
    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]['content']
        success = True
    else:
        result = response.text
        print("Request failed with status code:", response.status_code)
        print_with_color(response.text, "red")

    return success, result

def parse_label_rsp(rsp):
    try:
        # observation = re.findall(r"Observation:\s*(.*)", rsp, re.MULTILINE)[0]
        # think = re.findall(r"Thought:\s*(.*)", rsp, re.MULTILINE)[0]
        # title = re.findall(r"Title:\s*(.*)", rsp, re.MULTILINE)[0].strip()
        # act = re.findall(r"Action:\s*(.*)", rsp, re.MULTILINE)[0]
        # last_act = re.findall(r"Summary:\s*(.*)", rsp, re.MULTILINE)[0]
        observation = re.findall(r"观察：\s*(.*)", rsp, re.MULTILINE)[0]
        think = re.findall(r"思考：\s*(.*)", rsp, re.MULTILINE)[0]
        title = re.findall(r"标题：\s*(.*)", rsp, re.MULTILINE)[0].strip()
        act = re.findall(r"行动：\s*(.*)", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"总结：\s*(.*)", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Title:", "yellow")
        print_with_color(title, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        result = {
            "observation": observation,
            "think":think,
            "title":title,
            "action":"FINISH",
            "params":[],
            "summary":last_act         
        }
        params = []
        if "FINISH" in act:
            return result
        act_name = act.split("(")[0]
        result["action"] = act_name
        if act_name == "tap":
            if title and title != '无' and title != '空' and not '无文本' in title:
                params = [title]
                result["action"] = 'tap_text'
                print_with_color(f"converd to tap_text, title:{title}", "blue")
            else:
                area = int(re.findall(r"tap\((.*?)\)", act)[0])
                params = [area]
        elif act_name == "text":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            params = [input_str]
        elif act_name == "tap_text":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            params = [input_str]
        elif act_name == "long_press":
            area = int(re.findall(r"long_press\((.*?)\)", act)[0])
            params = [area]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0]
            area, swipe_dir = params.split(",")
            area = int(area)
            swipe_dir = swipe_dir.strip()[1:-1]
            params = [area, swipe_dir]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return None
        result["params"] = params
        return result
    except Exception as e:
        print_with_color(f"ERROR1: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return None