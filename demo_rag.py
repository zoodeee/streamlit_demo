# demo.py (수정된 버전)

# =================================================================
# 1. 라이브러리 및 환경 설정
# =================================================================
import os
import streamlit as st
import trafilatura
from openai import OpenAI
from dotenv import load_dotenv

# LangChain RAG 구성 요소
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings # Upstage 임베딩 모델

load_dotenv()

# =================================================================
# 2. RAG 시스템 초기화 (캐싱하여 한 번만 실행)
# =================================================================

# @st.cache_resource: 리소스가 크고 한 번만 로드되어야 할 때 사용
@st.cache_resource(show_spinner="RAG 시스템 초기화 중...")
def initialize_rag_system(api_key):
    """RAG에 필요한 임베딩 모델 로드, 문서 청킹, 벡터 DB를 초기화합니다."""
    # ... (2-1 문서 로딩 및 2-2 임베딩/DB 생성 코드는 그대로 유지)
    
    # 2-1. 문서 로딩 및 청킹
    data_dir = "./data" 
    docs = []

    if not os.path.isdir(data_dir):
        st.warning(f"경고: RAG 문서 디렉토리 '{data_dir}'를 찾을 수 없습니다. 일반 챗봇 모드로 실행됩니다.")
        return None

    for file in os.listdir(data_dir):
        if file.endswith(".html"):
            file_path = os.path.join(data_dir, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                text = trafilatura.extract(html_content)
                if text:
                    docs.append(Document(page_content=text, metadata={"source": file}))
            except Exception:
                continue
    
    if not docs:
        st.warning("경고: RAG를 위한 HTML 문서가 'data' 디렉토리에서 발견되지 않았습니다. 일반 챗봇 모드로 실행됩니다.")
        return None 

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # 2-2. 임베딩 모델 로드 및 벡터 DB 생성
    try:
        embedding_model = UpstageEmbeddings(upstage_api_key=api_key, model="embedding-query")
        
        db = Chroma.from_documents(
            documents=split_docs,
            embedding = embedding_model,
            persist_directory="./chroma_db", 
            collection_name = "Creation_evolution"
        )
        
        retriever = db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        return retriever
        
    except Exception as e:
        st.error(f"Upstage 임베딩 모델 또는 Chroma DB 초기화 오류:")
        st.exception(e)
        return None


# =================================================================
# 3. Streamlit 클라이언트 및 세션 설정
# =================================================================

st.title("창조과학 챗봇: 비봇")

# API 키 가져오기
try:
    UPSTAGE_API_KEY = st.secrets["UPSTAGE_API_KEY"]
except KeyError:
    st.error("오류: Streamlit secrets에 'UPSTAGE_API_KEY'가 설정되어 있지 않습니다.")
    st.stop()

# Upstage 클라이언트 초기화 (OpenAI 호환 API)
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1" 
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url=UPSTAGE_BASE_URL 
)

# 세션 상태 설정
if "upstage_model" not in st.session_state:
    st.session_state["upstage_model"] = "solar-pro2"

if "messages" not in st.session_state:
    st.session_state.messages = []


# RAG 시스템 초기화 실행
RETRIEVER = initialize_rag_system(UPSTAGE_API_KEY)
if RETRIEVER is None:
    RAG_ENABLED = False
    st.info("RAG 기능이 비활성화되었습니다. 일반 챗봇 모드로 실행됩니다.")
else:
    RAG_ENABLED = True
    st.success("RAG 시스템 초기화 완료.")

# =================================================================
# 4. 채팅 인터페이스 및 RAG/일반 챗봇 로직 (핵심 수정)
# =================================================================

# 이전 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문할 내용을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    # 4-1. 기본 메시지 리스트 구성 (이전 대화 기록)
    final_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    
    # 4-2. RAG 검색 및 시스템 명령 삽입
    use_rag = False
    
    if RAG_ENABLED:
        try:
            results = RETRIEVER.invoke(prompt)
            retrieved_context = "\n---\n".join([doc.page_content for doc in results])
            
            # 검색 결과가 충분하고 유효할 때 (50자 이상)
            if retrieved_context and len(retrieved_context) > 50:
                use_rag = True
                
                # RAG 프롬프트 구성: 유연한 시스템 명령
                # RAG에 갇히지 않도록 "컨텍스트를 사용해라" 대신 "컨텍스트를 참고해라"로 변경
                rag_system_instruction = f"""
                당신은 창조과학을 연구하는 교수이며, 모든 답변은 한국어로 해야 합니다.
                사용자의 질문에 대해 당신의 일반적인 지식을 사용하되, 만약 아래의 [참고 컨텍스트]가 관련이 있다면 답변에 적극적으로 활용하여 답변의 깊이와 정확성을 높이세요.
                
                [참고 컨텍스트]
                {retrieved_context}
                """
                
                # 기존 메시지 리스트의 맨 앞에 System Message 추가
                final_messages.insert(0, {"role": "system", "content": rag_system_instruction})
                st.info("RAG 문서가 참조되었습니다.")

            else:
                st.info("RAG 문서에 관련 자료가 없어 일반적인 대답을 합니다.")
                
        except Exception:
            st.info("RAG 검색 오류가 발생하여 일반적인 대답을 합니다.")
            # 오류 발생 시 시스템 메시지 없이 이전 대화만 전달

    
    # 4-3. LLM 호출 및 응답 처리
    with st.chat_message("assistant"):
        
        # RAG를 사용하지 않을 경우, 일반적인 시스템 메시지 추가 (선택 사항)
        if not use_rag and not any(m["role"] == "system" for m in final_messages):
            general_system_instruction = "당신은 창조과학을 연구하는 교수입니다. 모든 답변은 한국어로 해 주세요."
            final_messages.insert(0, {"role": "system", "content": general_system_instruction})

        stream = client.chat.completions.create(
            model=st.session_state["upstage_model"],
            messages=final_messages, # RAG 여부에 따라 시스템 메시지가 다르게 구성됨
            stream=True,
        )
        # 응답을 스트리밍하며 출력
        response = st.write_stream(stream)
            

    # 최종 응답을 세션 상태에 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
