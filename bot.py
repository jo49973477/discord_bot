from google import genai
import asyncio
from openai import AsyncOpenAI
import os
import time

import discord
from discord.ext import commands

import hydra
from omegaconf import DictConfig, OmegaConf
from config import MainConfig

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from google.genai.types import HarmCategory, HarmBlockThreshold

import traceback

os.environ["USER_AGENT"] = "TakanashiKiara"


def build_brain(cfg):

    loader = WebBaseLoader(cfg.site)
    documents = loader.load()

    # B. 텍스트 쪼개기 (너무 길면 검색이 어려우니 문단 단위로 자름)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       
        chunk_overlap=200,     
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""] 
    )
    splits = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=cfg.client
    )
    
    batch_size = 5
    vectorstore = None
    for i in range(0, len(splits), batch_size):
        batch = splits[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(splits)//batch_size)+1}...")
        
        if vectorstore is None:
            # 첫 번째 배치는 벡터 저장소를 생성
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            # 두 번째부터는 기존 저장소에 추가
            vectorstore.add_documents(batch)
            
        time.sleep(2)  # 2초 휴식 (API 과부하 방지)

    print("✅ 임베딩 완료!")
    return vectorstore.as_retriever()


def build_grok_brain(cfg):

    loader = WebBaseLoader(cfg.site)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(documents)

    # [수정] 구글 대신 로컬 임베딩 사용!
    # 이 모델은 한국어 검색에 특화되어 있고, 내 컴퓨터 CPU로 돌아갑니다. (API 키 필요 없음)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # 나머지는 동일
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever


def main():

    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="config")
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    main_cfg = MainConfig(**raw_config)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, # 성적 콘텐츠 제한 해제
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,        # 괴롭힘/욕설 제한 해제
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,       # 혐오 발언 제한 해제
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, # 위험한 콘텐츠 제한 해제
    }

    os.environ["GOOGLE_API_KEY"] = cfg.client

    if main_cfg.model.startswith("gemma") or main_cfg.model.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(
            model=main_cfg.model, temperature=0.7, api_key=main_cfg.client,
            safety_settings = safety_settings
        )
        retriever = build_brain(main_cfg)

    else:
        llm = ChatGroq(
            model=main_cfg.model,
            temperature=0.7,
            api_key=main_cfg.client,  # (당연히 config.yaml로 빼는 게 좋겠죠?)
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
            "context": retriever,  # 검색해서 채우기
            "question": RunnablePassthrough(),  # 사용자의 말 그대로 채우기
            "prompt_persona": lambda x: (
                main_cfg.prompt
            ),  # ★ 고정된 설정값 넣기 (람다 사용!)
        }
        # [2단계: 요리]
        | prompt_template
        # [3단계: 서빙]
        | llm
        | StrOutputParser()
    )

    # 2. 디스코드 봇 설정
    intents = discord.Intents.default()
    intents.message_content = True  # 메시지 읽기 권한
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        activity = discord.Game(name="closing the case... 🔎", type=1)
        await bot.change_presence(status=discord.Status.online, activity=activity)
        print(
            f"🔎Test test, Hello, {bot.user.name}, Amelia Watson! #1 Detective at your service!🔍"
        )

    @bot.event
    async def on_message(message):
        # 봇이 쓴 메시지는 무시
        if message.author == bot.user:
            return

        content_lower = message.content.lower()  # 소문자로 변환해서 검사
        keywords_tuple = tuple(main_cfg.keywords)

        should_respond = (
            content_lower.startswith(keywords_tuple)
            or bot.user.mentioned_in(message)
            or isinstance(message.channel, discord.DMChannel)
        )

        if should_respond:
            async with message.channel.typing():
                # 호출 키워드 제거
                clean_text = message.content
                for k in keywords_tuple:
                    if clean_text.lower().startswith(k):
                        clean_text = clean_text[len(k) :].strip()
                        break

                # ★ 여기가 핵심! 에러 처리 구간 시작 ★
                try:
                    # 1. RAG 체인 실행 (동기 함수를 비동기로 실행해서 봇 멈춤 방지)
                    # (만약 asyncio.to_thread가 어렵다면 그냥 response = rag_chain.invoke(clean_text)로 해도 됨)
                    response = await asyncio.to_thread(rag_chain.invoke, clean_text)

                    # 2. 정상적인 답변 전송
                    await message.reply(response)

                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Error occurred: {error_msg}")
                    print("============ 🚨 에러 발생 로그 🚨 ============")
                    traceback.print_exc()
                    print("==============================================")

                    # (1) API 한도 초과 (Rate Limit / 413 / 429)
                    if (
                        "413" in error_msg
                        or "rate_limit" in error_msg.lower()
                        or "quota" in error_msg.lower()
                    ):
                        await message.reply(
                            "🕵️‍♀️ [Amelia is exhausted!] \n"
                            "Teamate, I consumed all quota of my stamina today!💦\n"
                            "Come here tomorrow, or request my assistant to pay some (ice) tea for me, hic!"
                        )

                    # (2) 그 외 알 수 없는 에러
                    else:
                        await message.reply(
                            f"🕵️‍♀️ [Time-Space error occurred]\n"
                            f"Please report this error to my detective office assistant, Yeongyoo Jo!😵‍💫\n"
                            f"```\n{error_msg}\n```"
                        )

    bot.run(main_cfg.discord_token)


if __name__ == "__main__":
    main()
