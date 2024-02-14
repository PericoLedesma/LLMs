import streamlit as st
import random
import time
import os

# Local imports
from local_models import *
from document_processing import *

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# ----------------- Streamlit App ----------------- #
def main(model_type='local'):
    st.set_page_config(page_title="Generate Blogs",
                       page_icon='🤖',
                       layout='centered',
                       initial_sidebar_state='expanded',  # collapsed
                       menu_items={
                           'Get Help': 'https://www.extremelycoolapp.com/help',
                           'Report a bug': "https://www.extremelycoolapp.com/bug",
                           'About': "# This is a header. This is an *extremely* cool app!"
                       }
                       )


    st.title("Chatbot title 🤖")
    # st.write("Description.")
    st.markdown("---")

    # # ---------- DATA PIPELINE ---------- #
    # if "document_pipeline" not in st.session_state:
    #     st.session_state.document_pipeline = DocumentPipeline(HF_API_TOKEN=HUGGINGFACEHUB_API_TOKEN,
    #                                                           OPENAI_KEY=OPENAI_API_KEY,
    #                                                           data_dir_path="../../data/genai/documents",
    #                                                           db_dir_path="../../data/genai/db",
    #                                                           archive_path='../../data/archive')
    #
    #     st.session_state.document_pipeline.create_db_document(split_type="token",
    #                                                           chunk_size=200,
    #                                                           embedding_type="HF",  # Embedding model (HF, OPENAI).
    #                                                           chunk_overlap=10,  # Overlap size between chunks.
    #                                                           vectorstore_type="CHROMA")  # vector storage (FAISS, CHROMA, SVM). Just CHROMA for now.

    # ---------- MODEL  ---------- #
    if model_type == 'local' and "model" not in st.session_state:
        st.session_state.model = localLLMmodel(filename='prompts/template.txt')

    # ---------- CHAT HISTORY ---------- #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------- Setup chat message history ----------  #
    with st.chat_message("ai"):  # "user", "assistant", "ai", "human"
        st.write("Hello 👋")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------- Chat logic ----------  #
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # relevant_docs = st.session_state.document_pipeline.do_question(question=prompt,
        #                                                           relevant_docs=None,
        #                                                           with_score=False)

        response = st.session_state.model.llm_response(prompt)
        st.session_state.messages.append({"role": "ai", "content": response})

        with st.chat_message("ai"):
            st.markdown(response)
            # st.write(response)

        # question = "How are you?"
        # print('Question asked:', question)
        # res = obj.do_question(question=question,
        #                     repo_id="declare-lab/flan-alpaca-large",
        #                     chain_type="stuff",
        #                     relevant_docs=None,
        #                     with_score=False,
        #                     temperature=0,
        #                     max_length=300,
        #                     language="Spanish"):language="ENGLISH")
        # print(res)


if __name__ == "__main__":
    print('=' * 80)
    print(' ' * 30, '\t Running App....', ' ' * 30)
    print('=' * 80)
    main()

    # st.header("Blog options", divider='rainbow')
    # st.header('_Streamlit_ is :blue[cool] :sunglasses:')
