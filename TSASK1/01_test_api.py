"""
第 1 步最小示例：只验证能不能调通大模型 API
不读 PDF、不做抽取，只发一句最简单的话，看能不能收到回复。
"""

import os
from openai import OpenAI

# 1. 从环境变量读取 API Key（不要把 Key 直接写在代码里）
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise SystemExit(
        "未检测到 DASHSCOPE_API_KEY。\n"
        "请先在终端执行：export DASHSCOPE_API_KEY=你的完整Key"
    )

# 2. 创建客户端：base_url 填百炼业务空间的兼容接口地址
#    如果页面给的专属域名是 ...maas.aliyuncs.com/compatible-mode/v1，
#    就把下面这行替换成页面复制到的完整 Base URL。
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 3. 调用对话接口：把消息发给模型，拿回回复
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "用一句话告诉我：API 调用已经成功。"},
    ],
    temperature=0.0,
)

# 4. 从返回结果里取出模型回复的文字
reply = response.choices[0].message.content

print("=== 模型返回 ===")
print(reply)
#（注：内容由AI生成）
