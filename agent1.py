from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import faiss
from langchain_core.runnables import RunnableParallel, RunnableLambda , RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.tools import tool
import requests
from urllib.parse import urlparse, parse_qs  

load_dotenv()

def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get("v", [None])[0]
youtube_url= input("Enter youtube video url:")
video_id = get_video_id(youtube_url)
if not video_id:
    print("Invalid YouTube URL. Please try again.")
    exit()

try:
    transcript_list1 = YouTubeTranscriptApi()
    transcript_list = transcript_list1.fetch(video_id)
    transcript = " ".join([t.text for t in transcript_list])
except TranscriptsDisabled:
    print("No captions available for this video.")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text=text_splitter.create_documents([transcript])
emmbedings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
vector_store = FAISS.from_documents(text, emmbedings)
reteiver=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
llm= ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2,google_api_key=os.getenv("GOOGLE_API_KEY"))
prompt=PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. keep the answer concise.
    Question: {question} 
    Context: {context}""",
    input_variables=["context", "question"]
)
def format_docs(retieved_docs):
    context_text="\n\n".join(docs.page_content for docs in retieved_docs)
    return context_text
parallel_chain=RunnableParallel({
    "context": reteiver| RunnableLambda(format_docs),
    "question" : RunnablePassthrough()
})
parser=StrOutputParser()
st=input("")
chain1=parallel_chain|prompt|llm|parser
print(chain1.invoke(st))
