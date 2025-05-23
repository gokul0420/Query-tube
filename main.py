from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from transformers import pipeline
from datasets import Dataset
from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_video_id(url):
    video_id_match = re.search(r'(?:youtu\.be/|youtube\.com/watch\?v=|/embed/|/v/|/e/|watch\?v=|youtu\.be/|embed/|v=)([^&#?]+)', url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        return None

youtube_url = "https://www.youtube.com/watch?v=q-_ezD9Swz4"
video_id = extract_video_id(youtube_url)

data = []

if video_id:
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB'])
    except:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

concatenated_data = []
chunk_size = 10
for i in range(0, len(srt), chunk_size):
    chunk = ' '.join(entry['text'] for entry in srt[i:i + chunk_size])
    concatenated_data.append({'text': chunk})

formatted_data = [{'page_content': entry['text']} for entry in concatenated_data]
dataset = Dataset.from_dict({"page_content": [entry['page_content'] for entry in formatted_data]})

data = [{'text': entry['page_content']} for entry in dataset]
docs = [Document(page_content=entry['text']) for entry in data]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(docs)

modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 4})

question = "is it really important for coding and if it is then explain why "

prompt = """
You are an AI assistant named YT Tutor. Use the provided context to answer the question in detail. Provide examples from the context if you cant find any example from the context then answer from your own. Ensure clarity and depth.
Context:
"""

pipe = pipeline("text-generation", model="Qwen/Qwen1.5-0.5B")  # add device param if GPU available

context = retriever.get_relevant_documents(question)

context_text = "\n".join([doc.page_content for doc in context])
full_prompt = prompt + "\n" + context_text + f"\n\nQuestion: {question}\nAnswer:"

result = pipe(full_prompt, temperature=0.7, do_sample=True)
answer = result[0]['generated_text'].split("Answer:")[1].strip()

print(answer)
