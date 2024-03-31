import streamlit as st
from streamlit_chat import message
from core import run_llm

def create_sources_string(source_urls) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

st.title("Document Helper")

with st.form("Document Helper"):
    user_message = st.text_area("質問を書いてください")
    submitted = st.form_submit_button("送信する")

    if submitted:
        st.session_state.past.append(user_message)

        generated_response = run_llm(query=user_message)
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state.generated.append(formatted_response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + "_user")
            message(st.session_state['generated'][i], key=str(i))