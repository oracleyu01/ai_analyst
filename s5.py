import streamlit as st
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import re

# 페이지 설정
st.set_page_config(page_title="울산 산불 분석 챗봇", page_icon="🔥")

# 사이드바 설정
st.sidebar.title("설정")
api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요", type="password")

# 컬렉션 이름 입력 (사이드바)
collection_name = st.sidebar.text_input("사용할 컬렉션 이름", value="ulsan_forestfire_202503231")

# 페이지 제목
st.title("🔥 울산 산불 분석 챗봇")
st.write("뉴스 기사 및 보고서에서 수집한 울산 산불 데이터에 대해 질문해보세요.")

# ChromaDB 클라이언트 초기화
@st.cache_resource
def init_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

# 벡터 데이터베이스에서 컬렉션 가져오기
def get_collection(collection_name):
    try:
        client = init_chroma_client()
        collection = client.get_collection(name=collection_name)
        return collection
    except Exception as e:
        st.sidebar.error(f"컬렉션 가져오기 오류: {e}")
        return None

# 벡터 데이터베이스 검색 함수
def search_vector_db(collection, query, n_results=3):
    try:
        if not collection:
            return [{"content": "컬렉션을 불러올 수 없습니다. 컬렉션 이름을 확인하세요.", "title": "오류", "metadata": {}}]
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = []
        for i in range(len(results['documents'][0])):
            document = {
                "content": results['documents'][0][i],
                "title": results['metadatas'][0][i].get('title', '제목 없음'),
                "metadata": results['metadatas'][0][i]
            }
            documents.append(document)
        
        return documents
    except Exception as e:
        st.sidebar.error(f"검색 오류: {e}")
        return [{"content": f"검색 중 오류 발생: {e}", "title": "오류", "metadata": {}}]

# OpenAI를 활용한 응답 생성 함수
def get_gpt_response(query, search_results, api_key, model="gpt-4o-mini"):
    if not api_key:
        return "OpenAI API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력해주세요."

    try:
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=api_key)
        
        # 컨텍스트 구성
        context = "다음은 뉴스 기사와 보고서에서 수집한 울산 산불 관련 데이터입니다:\n\n"
        
        for i, result in enumerate(search_results):
            context += f"문서 {i+1}:\n"
            context += f"제목: {result['title']}\n"
            
            # 메타데이터 추가
            if result['metadata']:
                metadata = result['metadata']
                if 'date' in metadata:
                    context += f"작성일: {metadata['date']}\n"
                if 'location' in metadata:
                    context += f"발생지역: {metadata['location']}\n"
                if 'damage_scale' in metadata:
                    context += f"피해규모: {metadata['damage_scale']}\n"
            
            # 내용 요약 (너무 길면 잘라냄)
            content = result['content']
            if len(content) > 800:
                content = content[:800] + "..."
            context += f"내용: {content}\n\n"
        
            # 개선된 GPT 프롬프트
            system_prompt = """당신은 산불 및 재난 분석 전문가입니다. 산불 발생 원인과 확산 과정을 정확하게 파악하는 능력이 뛰어납니다. 
            제공된 문서들을 바탕으로 특히 '울산 산불 발생 원인'에 초점을 맞추어 분석해 주세요.

            답변 작성 가이드라인:
            1. 문서에서 울산 산불의 직접적인 발생 원인과 배경을 먼저 찾아 강조해 주세요.
            2. 산불 발생 지역, 시간대, 기상 조건을 명확히 설명해 주세요.
            3. 초기 대응, 확산 경로, 진화 과정 등 산불 진행 상황을 구체적으로 분석해 주세요.
            4. 문서에 언급된 구체적인 데이터나 수치(피해면적, 인명피해 등)를 활용해 주세요.
            5. 산불 예방과 대응을 위한 교훈과 제안사항도 포함해 주세요.
            6. 사용자의 질문에 직접적으로 관련된, 산불 원인에 집중하세요."""

            user_prompt = f"""{context}

            사용자 질문: {query}

            위 문서들을 분석하여 '울산 산불의 발생 원인'에 초점을 맞춘 전문적인 답변을 제공해주세요.
            특히 문서 1에 집중하여 산불 발생의 주요 원인과 배경을 자세히 설명해주세요.
            다른 문서에서도 산불 원인과 관련된 내용이 있다면 함께 분석해주세요.

            답변에는 다음 내용을 반드시 포함해주세요:
            1. 울산 산불의 구체적인 발생 원인(가능한 원인들 모두 분석)
            2. 산불 확산에 영향을 미친 기상 조건이나 지형적 요인
            3. 지역별 특성이 산불 확산에 미친 영향
            4. 문서에 언급된 구체적인 수치나 데이터

            각 원인과 요인을 별도 단락으로 구분하고, 중요한 정보는 강조해서 표시해주세요."""
                    
        # API 호출
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=800
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        
        # 인증 오류 확인
        if "auth" in error_msg.lower() or "api key" in error_msg.lower():
            return "OpenAI API 키 인증에 실패했습니다. API 키를 확인해주세요."
        else:
            return f"분석 중 오류가 발생했습니다: {error_msg}"

# API 키가 없을 경우 간단한 응답 함수
def get_simple_response(query, search_results):
    if not search_results or search_results[0].get("title") == "오류":
        return "관련 데이터를 찾을 수 없습니다."
    
    result_text = f"'{query}'에 대한 검색 결과:\n\n"
    
    for i, result in enumerate(search_results):
        result_text += f"**문서 {i+1}:** {result['title']}\n"
        
        # 내용 요약 (100자로 제한)
        content = result['content']
        if len(content) > 100:
            content = content[:100] + "..."
        result_text += f"{content}\n\n"
    
    result_text += "더 자세한 분석을 위해서는 OpenAI API 키를 입력해주세요."
    return result_text

# 챗봇 응답 생성 함수
def chat_response(question, collection):
    # 벡터 데이터베이스 검색
    search_results = search_vector_db(collection, question)
    
    # ChatGPT API 키가 있으면 GPT 사용, 없으면 간단한 응답
    if api_key:
        return get_gpt_response(question, search_results, api_key)
    else:
        return get_simple_response(question, search_results)

# 컬렉션 가져오기
collection = get_collection(collection_name)

# 컬렉션 정보 표시
if collection:
    try:
        count = collection.count()
        st.sidebar.success(f"컬렉션 '{collection_name}'에서 {count}개의 문서를 불러왔습니다.")
    except Exception as e:
        st.sidebar.warning(f"컬렉션 정보 확인 중 오류: {e}")
else:
    st.sidebar.warning(f"컬렉션 '{collection_name}'을 찾을 수 없습니다. 이름을 확인하세요.")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("질문을 입력하세요 (예: 울산 산불의 주요 원인은 무엇인가요?)"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 사용자 메시지 저장
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 응답 생성
    with st.spinner("답변 생성 중..."):
        response = chat_response(prompt, collection)

    # 응답 메시지 표시
    with st.chat_message("assistant"):
        st.markdown(response)

    # 응답 메시지 저장
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# 예시 질문
st.sidebar.header("예시 질문")
example_questions = [
    "울산 산불의 주요 원인은 무엇인가요?",
    "산불 확산을 가속화한 기상조건은?",
    "울산 산불의 피해 규모는 어느 정도인가요?",
    "산불 진화 과정에서의 어려움은 무엇이었나요?"
]

# 예시 질문 버튼
for question in example_questions:
    if st.sidebar.button(question):
        # 사용자 메시지 표시 및 저장
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # 응답 생성
        with st.spinner("답변 생성 중..."):
            response = chat_response(question, collection)

        # 응답 메시지 표시 및 저장
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # 페이지 새로고침
        st.rerun()

# 대화 기록 초기화 버튼
if st.sidebar.button("대화 기록 초기화"):
    st.session_state.chat_history = []
    st.rerun()

# 컬렉션 목록 표시
try:
    client = init_chroma_client()
    collections = client.list_collections()
    
    with st.sidebar.expander("사용 가능한 컬렉션 목록"):
        for coll in collections:
            st.write(f"- {coll}")
except Exception as e:
    st.sidebar.error(f"컬렉션 목록 로드 오류: {e}")