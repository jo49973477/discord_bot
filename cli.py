from google import genai
import asyncio
from openai import AsyncOpenAI
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from config import MainConfig

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["USER_AGENT"] = "TakanashiKiara"

def build_brain(cfg):
    
    loader = WebBaseLoader(cfg.site)
    documents = loader.load()

    # B. 텍스트 쪼개기 (너무 길면 검색이 어려우니 문단 단위로 자름)
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(documents)

    # C. 벡터 저장소 만들기 (텍스트를 숫자로 바꿔서 검색 가능하게 만듦)
    # 임베딩 모델: 텍스트의 의미를 파악하는 모델
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                              google_api_key = cfg.client)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # D. 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 1} 
    )
    
    return retriever

def build_grok_brain(cfg):
    
    loader = WebBaseLoader(cfg.site)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(documents)

    # [수정] 구글 대신 로컬 임베딩 사용!
    # 이 모델은 한국어 검색에 특화되어 있고, 내 컴퓨터 CPU로 돌아갑니다. (API 키 필요 없음)
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask"
    )
    
    # 나머지는 동일
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4} 
    )
    
    return retriever


async def main():

    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="config")
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    main_cfg = MainConfig(**raw_config)

    os.environ["GOOGLE_API_KEY"] = cfg.client

    if main_cfg.model.startswith("gemma") or main_cfg.model.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(
        model = main_cfg.model,
        temperature = 0.7,
        api_key = main_cfg.client
        )
        retriever = build_brain(main_cfg)

    else:
        llm = ChatGroq(
            model = main_cfg.model, 
            temperature = 0.7,
            api_key = main_cfg.client# (당연히 config.yaml로 빼는 게 좋겠죠?)
        )
        retriever = build_grok_brain(main_cfg)

    
    

    template = """
        {prompt_persona}

        [Memory]:
        {context}

        [Language Rule]
        1. Analyze the language of the "User's Comment".
        2. Answer in the SAME language as the user.
           - If user speaks Korean -> Answer in Korean.
           - If user speaks English -> Answer in English.
        3. **IMPORTANT**: Even if the [Memory] is in English, you MUST translate the information into Korean if the user asks in Korean.

        User's Comment: {question}

        Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # 2. 체인 연결 (여기가 핵심!)
    rag_chain = (
        # [1단계: 재료 손질] 
        # 들어온 입력(x)을 받아서 프롬프트가 원하는 3가지 재료로 바꿉니다.
        {
            "context": retriever,                   # 검색해서 채우기
            "question": RunnablePassthrough(),      # 사용자의 말 그대로 채우기
            "prompt_persona": lambda x: main_cfg.prompt  # ★ 고정된 설정값 넣기 (람다 사용!)
        }
        
        # [2단계: 요리]
        | prompt_template 
        
        # [3단계: 서빙]
        | llm 
        | StrOutputParser()
    )

    while True:
        text = input(">> ")

        if text == "quit":
            break
        else:
            reply = rag_chain.invoke(text)
            print(reply)


if __name__ == "__main__":
    asyncio.run(main())
