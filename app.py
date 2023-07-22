import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import StdOutCallbackHandler
from dotenv import load_dotenv

import os



with st.sidebar:
    load_dotenv()
    st.title('ChatBot LLM Project')
    st.markdown('''
                ## About
                This App is an LLM-Powered chatbot built using:
                [StreamLit](https://streamlit.io/)
                [LangChain](https://python.langchain.com)
                [OpenAI](https://platform.openai.com/docs/models) LLM Model
                ''')
    add_vertical_space(5)
    st.write('Made by Fred Pi')


def main():
    st.header("PDF AI Chatbot")

    # -- upload pdf
    pdf = st.file_uploader("Choose a PDF file to upload:", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            # token size
            chunk_size=1000,
            # token overlap
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)        
        store_name = pdf.name[:-4]

        if os.path.existis(f"{store_name}.pk1", "wb"):
            with open(f"{store_name}.pk1", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loaded.")    

        else:
            ## -- create embeddings
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pk1", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write("Embeddings computation completed.")

        # -- accept queries
        query = st.text_input("Insert queries about loaded files.")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


            st.write(docs)


        st.write(chunks)


if __name__ == '__main__':
    main()
