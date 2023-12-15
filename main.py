# Cheon Groo / 010-7720-0085 / forest@handong.ac.kr
# Import necessary modules
import pandas as pd
import streamlit as st 
from PIL import Image
from PyPDF2 import PdfReader

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import *


home_privacy = "We value and respect your privacy. To safeguard your personal details, we utilize the hashed value of your OpenAI API Key, ensuring utmost confidentiality and anonymity. Your API key facilitates AI-driven features during your session and is never retained post-visit. You can confidently fine-tune your research, assured that your information remains protected and private."

st.set_page_config(
    page_title="GPT for Human & Forest",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # checking if page_text is not None or empty string
                text += page_text
            else:
                print(f"Failed to extract text from a page in {pdf}")

    return text

# def chunk_size():
#     global chunk_size
#     chunk_size = 1000
#     return chunk_size

# def chunk_overlap():
#     global chunk_overlap
#     chunk_overlap = 200
#     return chunk_overlap


def get_text_chunks(text):
    # text_splitter = RecursiveCharacterTextSplitter(
    #     separators="\n",
    #     chunk_size=chunk_size(),
    #     chunk_overlap=chunk_overlap(),
    #     length_function=len
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings(openai_api_key="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")
    embeddings = embeddings = HuggingFaceEmbeddings(
        model_name="TheBloke/Llama-2-7B-Chat-GGML", encode_kwargs={'normalize_embeddings': True})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain():
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    if model == 'gpt-3.5-turbo':
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=temperature_input, model_name=model_select),
            retriever=vectorstore.as_retriever(),
            get_chat_history=lambda h : h,
            memory=memory
        )
    if model == 'Llama2 7B':
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q2_K.bin", model_type="llama"),
            retriever=vectorstore.as_retriever(),
            get_chat_history=lambda h : h,
            memory=memory
        )
    return conversation_chain

def pdf_to_document(pdf_docs):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf_docs.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf_docs.getvalue())
    pdf_loader = PyPDFLoader(temp_filepath)
    pages = pdf_loader.load_and_split()
    return pages

def csv_to_document(csv_docs):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, csv_docs.name)
    with open(temp_filepath, "wb") as f:
        f.write(csv_docs.getvalue())
    csv_loader = CSVLoader(temp_filepath, csv_args={'delimiter': ',','quotechar': '"'})
    csv = csv_loader.load()
    return csv



st.sidebar.subheader("모델")
llm_model_options = ['Llama2 7B','gpt-3.5-turbo'] 
model_select = st.sidebar.selectbox('LLM 모델 선택 :', llm_model_options, index=0)
if model_select == 'Llama2 7B':
    model = 'Llama2-7B'
elif model_select == 'gpt-3.5-turbo':
    model = 'gpt-3.5-turbo'
else:
    model_select = 'Llama2-7B'
st.sidebar.markdown("""\n""")
# temperature_input = st.sidebar.slider('청크 사이즈 :', min_value=100, max_value=2000, value=1000, on_change=chunk_size)
# st.sidebar.slider('오버랩 :', min_value=0, max_value=500, value=200, on_change=chunk_overlap())
temperature_input = st.sidebar.slider('청크 사이즈 :', min_value=100, max_value=2000, value=1000)
st.sidebar.slider('오버랩 :', min_value=0, max_value=500, value=200)
st.sidebar.markdown("""\n""")
clear_history = st.sidebar.button("대화 기록 삭제")


with st.sidebar:
    st.divider()
    st.subheader("지원 파일 :", anchor=False)
    st.info(
        """
        .pdf, .csv

        """)

    st.divider()

st.sidebar.subheader("설정")
OPENAI_API_KEY = st.sidebar.text_input("Key : ", type="password")

with st.sidebar:
    #st.subheader("Cheon Groo", anchor=False)
    
    # st.subheader("🔗 Contact / Connect:", anchor=False)
    # st.markdown(
    #     """
    #     - [Email](mailto:yb.codes@gmail.com)
    #     - [LinkedIn](https://www.linkedin.com/in/yakshb/)
    #     - [Github Profile](https://github.com/yakshb)
    #     - [Medium](https://medium.com/@yakshb)
    #     """
    # )

    st.divider()
    st.write("Cheon Groo \n")
    st.write("With Llama2 / LangChain / Streamlit")


st.markdown(f"""## GPT for <span style=color:orange>Mediagroup Human & Forest</font></span>""",unsafe_allow_html=True)
st.write("_* Project [LLM Models Customized to Enterprise]_")

with st.expander("🔎 How To Use"):
    st.info("""
    1. **문서 업로드 및 프로세싱**: 파일 탐색 버튼을 클릭하여 문서 또는 경로를 선택한 후, 업로드 버튼을 통해 학습 데이터로 변환합니다.
    
    2. **대화형 AI**: 문서와 관련된 쿼리문을 작성해 질문하면, GPT 모델이 적절한 답변을 도출합니다.
    
    3. **답변 품질 파라미터 조정** : 답변의 청크 사이즈를 높이면, 비교적 긴 답변을 도출하지만 답변 속도가 느려집니다. 답변의 오버랩을 높이면, 문단간의 문맥의 흐름이 자연스러워지지만 답변의 내용이 겹치는 부분이 발생합니다.

    """)

# Upload file to Streamlit app for querying
user_uploads = st.file_uploader("파일 업로드", accept_multiple_files=True)

if user_uploads is not None:
    if st.button("학습 데이터로 추가"):
        with st.spinner("프로세싱"):
            if user_uploads[0].name.endswith('.csv'):
                csv = csv_to_document(user_uploads[0])
                df = pd.read_csv(csv)
                st.session_state.conversation = get_conversation_chain(df)
            if user_uploads[0].name.endswith('.pdf') :
                pdf_to_document(user_uploads[0])
                raw_text = get_pdf_text(user_uploads)
                # # st.write(raw_text)
                text_chunks = get_text_chunks(raw_text)
                # ## st.write(text_chunks) 
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

                # embeddings_model = OpenAIEmbeddings(openai_api_key="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")

            


user_query = st.chat_input(" 질문을 입력해주세요.")
st.session_state['doc_messages'] = [{"role": "user", "content": user_query}]
st.session_state['doc_messages'] = [{"role": "assistant", "content": user_query}]

st.session_state['chat_history'] = []


st.session_state['doc_messages'].append({"role": "user", "content": user_query})
with st.chat_message("user"):
    st.markdown(user_query)

with st.spinner("답변을 생성하는 중입니다..."):
    if 'conversation' in st.session_state:
        st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
            {
                "role": "user",
                "content": user_query
            }
        ]
        result = st.session_state.conversation({
            "question": user_query, 
            "chat_history": st.session_state['chat_history']})
        response = result["answer"]
        st.session_state['chat_history'].append({
            "role": "assistant",
            "content": response
        })
    else:
        response = "대화를 시작하기 전 문서를 먼저 업로드 해주세요."

    with st.chat_message("assistant"):
        st.write(response)
    st.session_state['doc_messages'].append({"role": "assistant", "content": response})
