from openai import OpenAI

import streamlit as st



# 1. 클라이언트 초기화 수정

st.title("창조과학 챗봇: 비봇")



# Upstage의 OpenAI 호환 API 엔드포인트 (실제 엔드포인트로 변경 필요)

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"



# st.secrets["UPSTAGE_API_KEY"]로 API 키 변경을 가정

client = OpenAI(

    api_key=st.secrets["UPSTAGE_API_KEY"],

    base_url=UPSTAGE_BASE_URL

)



# 2. 모델 이름 및 세션 변수 수정

if "upstage_model" not in st.session_state:

    # Upstage에서 제공하는 모델 이름으로 변경 (예시)

    st.session_state["upstage_model"] = "solar-pro2"



if "messages" not in st.session_state:

    st.session_state.messages = []



for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



if prompt := st.chat_input("What is up?"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):

        st.markdown(prompt)



    with st.chat_message("assistant"):

        stream = client.chat.completions.create(

            # 3. API 호출 시 모델 변수 사용

            model=st.session_state["upstage_model"],

            messages=[

                {"role": m["role"], "content": m["content"]}

                for m in st.session_state.messages

            ],

            stream=True,

        )

        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})