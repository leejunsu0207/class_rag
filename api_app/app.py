import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 환경 설정 및 세션 초기화
load_dotenv()
st.set_page_config(page_title="RAG 어시스턴트", page_icon="🤖")

if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

# 2. 리소스 캐싱 (함수화하여 성능 최적화)
@st.cache_resource
def get_retriever(file_path, file_name):
    # 노트북의 문서 로드 & 분할 로직
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    split_docs = text_splitter.split_documents(docs)

    # 임베딩 및 벡터 DB 생성
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever()

# 3. 사이드바 - 파일 업로드
st.sidebar.title("설정")
uploaded_file = st.sidebar.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    # 1. 고유한 임시 파일 생성
    # suffix=".pdf"를 주어 loader가 PDF임을 인식하게 합니다.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # 2. retriever 호출 (파일 경로와 함께 파일 이름을 넘겨 캐싱 효율을 높임)
    # 파일명(uploaded_file.name)을 인자로 넘겨야 파일이 바뀔 때 캐시가 갱신됩니다.
    retriever = get_retriever(tmp_path, uploaded_file.name)

    st.sidebar.success(f"'{uploaded_file.name}' 분석 완료!")

    # 임시 파일은 분석 후 삭제해도 벡터 DB는 메모리에 남아있습니다 (필요 시)
    # os.remove(tmp_path)

    # 4. 메인 채팅 UI
    st.title("📄 문서 기반 대화")

    # 기존 대화 표시
    for msg in st.session_state.history.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    # 사용자 입력 처리
    if prompt_input := st.chat_input("질문을 입력하세요"):
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # 체인 생성 (노트북의 프롬프트 개량판 적용)
        llm = ChatGoogleGenerativeAI(model="gemini-2-flash") # 2.5 대신 안정적인 버전 권장

        template = """당신은 유능한 비서입니다. 이전 대화 기록과 문맥을 바탕으로 답하세요.
        
        이전 대화: {chat_history}
        문맥: {context}
        질문: {question}
        답변:"""

        prompt = PromptTemplate.from_template(template)

        chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: st.session_state.get("history").messages if st.session_state.get("history") else []
                }
                | prompt
                | llm
                | StrOutputParser()
        )

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_answer = ""
            for chunk in chain.stream(prompt_input):
                full_answer += chunk
                response_container.markdown(full_answer + "▌")
            response_container.markdown(full_answer)

        # 대화 내용 저장
        st.session_state.history.add_user_message(prompt_input)
        st.session_state.history.add_ai_message(full_answer)
else:
    st.info("왼쪽 사이드바에서 PDF 파일을 먼저 업로드해주세요.")