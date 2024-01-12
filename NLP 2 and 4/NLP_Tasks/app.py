
#pip uninstall -r requirements.txt
#pip install -r requirements.txt
#streamlit run app.py



import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai

st.set_page_config(page_title='793_final_project_NLP')

openai_key="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"
os.environ["OPENAI_API_KEY"] = openai_key

#*********************************************************************#
#Task 2
#*********************************************************************#

def NLP_PDFChatBox():
    st.header("A PDF reader")
    st.write("This app uses OpenAI's LLM model to answer questions about your PDF file. Upload your PDF file and ask questions about it. The app will return the answer from your PDF file.")


    st.header("1. Upload PDF")
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
                
        # Accept user questions/query
        st.header("2. Ask questions about your PDF file:")
        q="Tell me about the content of the PDF"
        query = st.text_input("Questions",value=q)
        # st.write(query)

        if st.button("Ask"):
            if openai_key=='':
                st.write('Warning: Please pass your OPEN AI API KEY on Step 1')
            else:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.header("Answer:")
                st.write(response)
                st.write('--')
                st.header("OpenAI API Usage:")
                st.text(cb)



#*********************************************************************#
#Task 4
#*********************************************************************#
def ChatGPT(user_query):
    ''' 
    This function uses the OpenAI API to generate a response to the given 
    user_query using the ChatGPT model
    '''
    # Use the OpenAI API to generate a response
    

    model_engine = "text-davinci-003"
    completion = openai.Completion.create(
                                  engine = model_engine,
                                  prompt = user_query,
                                  max_tokens = 1024,
                                  n = 1,
                                  temperature = 0.5,
                                      )
    response = completion.choices[0].text
    return response

    
def NLP_LimtScope():
    st.header("Limit the answer area")
    st.write("Create an interface that limits a language model to answer only specific types of questions. For example, it can only ask questions about sports.")

    #openai.api_key = os.getenv("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")
    openai.api_key = "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"
    
    st.header("1. Enter a specific area's name ")
    a = "sport"
    area_query = st.text_input("Enter here",key = a, label_visibility="hidden")
    
    st.header("2. Enter your question ")
    b = "do you like some apple?"
    user_query = st.text_input("Enter here",key=b, label_visibility="hidden")
    area_limit = user_query +  " Is the question above is  a " + area_query + " question? Answer with yes or no. Please do not include any punctuation in your answer. Thank you."
    

    
    if area_limit != ":q" or area_limit != "":
        # Pass the query to the ChatGPT function
        response = ChatGPT(area_limit) #str
        # st.write(f"{area_limit} {response}")
        # print(is_yes(response)) 
        # print(user_query!= ":q")
    
    
    if 'yes' in response.lower():
        final_response = ChatGPT(user_query)
        
    else:        
        final_response = "Sorry, the question is not a " + area_query +" question."        
    
    st.write(f"{final_response}")
        



#*********************************************************************#
#Side Bar
#*********************************************************************#

with st.sidebar:
    st.title('Shiyu Wang')
    st.title('NLP Projects for CS793')
    st.markdown('''
    ## About
    I merge the two NLP tasks to this website. Please choose a task to enter the relevent page.
    ''')
    
    page = st.sidebar.selectbox('', 
                                ['NLP_PDFChatBox', 
                                 'NLP_LimtScope'])

    add_vertical_space(2)
    st.write('## References')
    st.markdown('''
                **NLP_Task 2: NLP_PDF ChatBox**  
                [1] [Livia Ellen's github](https://python.plainenglish.io/create-your-own-chatbot-for-pdf-documents-with-openai-gpt-and-streamlit-e5b35826bc1e')  
                [2] [Error Fix](https://community.openai.com/t/attributeerror-module-openai-has-no-attribute-error/486676)  
                [3] [sai harish cherukuri's blog](https://saiharishcherukuri.medium.com/pdf-summarizer-and-question-answering-unlocking-insights-from-pdf-documents-f8933620b1c4)  
                  
                **NLP_Task 4: NLP_Limit answer area**  
                [1] [Yasser Mustafa's blog](https://blog.devgenius.io/building-a-chatgpt-web-app-with-streamlit-and-openai-a-step-by-step-tutorial-1cd57a57290b)
                ''')
    
    # 根据用户的选择显示不同的界面
if page == 'NLP_PDFChatBox':
    NLP_PDFChatBox()
elif page == 'NLP_LimtScope':
    NLP_LimtScope()


# if __name__ == '__main__':
#     main()