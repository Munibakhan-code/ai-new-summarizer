# from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_result = TavilySearchResults(max_results=3)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assisstant

    summarize the following news into clear bullet points.

    {news}
    """
)

chain = prompt | llm | StrOutputParser()

news_result = search_result.invoke("tell me about the tech job in Augest 2026")
print("========================================")
print(news_result)
print("========================================")

final_result = chain.invoke({"news": news_result})

print(final_result)