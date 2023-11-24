import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import  HuggingFaceEmbeddings
from langchain.vectorstores import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
import pinecone as pn

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '54b98367-a331-4be8-bed7-bada25e79d91')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

def get_pdf_text(pdf):
    text = ""
    loader = PyPDFLoader("budget_speech.pdf")
    # pdf_reader = PdfReader(pdf)
    data = loader.load()
    # for page in pdf_reader.pages:
    #    text += page.extract_text()
    return data


def get_text_chunks(text):
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs=text_splitter.split_documents(text)
    # chunks = text_splitter.split_text(text)
    return docs


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    # embeddings = HuggingFaceEmbeddings(
    # model_name=model_name,
    # model_kwargs=model_kwargs,
    # encode_kwargs=encode_kwargs
    # )
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # initialize pinecone
    pn.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "second" # put in the name of your pinecone index here
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore=pinecone.Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
    # vectorstore = pinecone.Pinecone.from_existing_index("second", embeddings)
    # vectorstore = FAISS.from_texts(texts=, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # clean pdf for better results
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})    
    st.write(response)
    
    # st.session_state.chat_history = response['chat_history']

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with PDF",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "hello human"),unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "hello robot"),unsafe_allow_html=True)
    

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'", type=["pdf"])
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write(vectorstore)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                     vectorstore)


if __name__ == '__main__':
    main()