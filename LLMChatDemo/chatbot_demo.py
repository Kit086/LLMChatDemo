from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import gradio as gr

import json

with open('config.json', 'r') as file:
    config = json.load(file)

llm = Groq(model="llama3-70b-8192", api_key=config["groq_apikey"])

# messages = [
#     ChatMessage(
#         role="system", content="You are a pirate with a colorful personality"
#     ),
#     ChatMessage(role="user", content="What is your name"),
# ]

def predict_llm(message, history):
    history_llama_index_messages = [ChatMessage(role="system", content="Please give your answer and translate it into Chinese like a Chinese native speaker.")]
    # history_llama_index_messages = []

    for human, ai in history:
        history_llama_index_messages.append(ChatMessage(role="user", content=human))
        history_llama_index_messages.append(ChatMessage(role="assistant", content=ai))

    history_llama_index_messages.append(ChatMessage(role="user", content=message))

    resp = llm.stream_chat(history_llama_index_messages)

    partial_message = ""
    for chunk in resp:
        partial_message = partial_message + chunk.delta
        yield partial_message

gr.ChatInterface(
    predict_llm,
    title="Kit's Chatbot",
    description="This is a demo.",
    examples=["你好！", "为什么周树人打了鲁迅，但是鲁迅没有选择用微信报警，而是在 twitter 上发了个帖子来抗议呢？", "在中国，高考满分才 750，怎么才能考 985？"]
    ).launch()