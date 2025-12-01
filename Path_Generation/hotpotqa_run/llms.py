import os
import openai
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from Path_Generation.hotpotqa_run.config import OPENAI_API_KEY
import tiktoken
token_enc = tiktoken.get_encoding("cl100k_base")

OPENAI_CHAT_MODELS = ["gpt-3.5-turbo","gpt-3.5-turbo-16k-0613","gpt-3.5-turbo-16k","gpt-4-0613","gpt-4-32k-0613"]
FASTCHAT_MODELS = ["llama-2-13b-chat","vicuna-7b"]
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


class langchain_openai_chatllm:
    def __init__(self, llm_name):
        openai.api_key = OPENAI_API_KEY
        self.llm_name = llm_name
        human_template = "{prompt}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
   
    def run(self, prompt, temperature=1, stop=['\n'], max_tokens=128):
        chat = ChatOpenAI(model=self.llm_name, temperature=temperature, stop=stop, max_tokens=max_tokens)
        self.chain = LLMChain(llm=chat, prompt=self.chat_prompt)
        return self.chain.run(prompt)


class langchain_fastchat_llm:
    def __init__(self, llm_name):
        openai.api_key = "EMPTY"  # Not supported yet
        self.prompt_temp = PromptTemplate(
            input_variables=["prompt"], template="{prompt}"
        )
        self.llm_name = llm_name
        
    def run(self, prompt, temperature=0.9, stop=['\n'], max_tokens=128):
        openai.api_base = "http://10.1.50.26:8000/v1"
        llm = OpenAI(
            model=self.llm_name, 
            temperature=temperature, 
            top_p=0.75, 
            top_k=40, 
            num_beams=4, 
            stop=stop,
            max_tokens=max_tokens
        )
        chain = LLMChain(llm=llm, prompt=self.prompt_temp)
        return chain.run(prompt)


def get_llm_backend(llm_name):
    if llm_name in OPENAI_CHAT_MODELS:
        return langchain_openai_chatllm(llm_name)
    else:
        return langchain_fastchat_llm(llm_name)
