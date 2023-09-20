import pandas as pd 
from bs4 import BeautifulSoup
import langchain
from langchain.llms import VertexAI
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document 
from langchain.chains import VectorDBQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
# from vertexai.preview.language_models.language_models import ChatModel, InputOutputTextPair

# TEXT_SPLITTER = RecursiveCharacterTextSplitter(
#     chunk_size=500, chunk_overlap=0, length_function=len, separators=[" ", ",", "\n", "."]
# )

EMBEDDINGS = VertexAIEmbeddings()

LLM = VertexAI(model_name='text-bison@001', max_output_tokens='800', temperature=0)




def get_db(class_name):

    persist_directory = f'vector_databases/{class_name}_db'

    vectordb = Chroma(persist_directory=persist_directory, 
                      embedding_function=EMBEDDINGS)


    return vectordb

metadata_field_info = [
    AttributeInfo(
        name = 'url',
        description='Where the document is from. Should be a link to the document',
        type = 'string'
    ),
    AttributeInfo(
        name = 'tag',
        description= "What subc-class the document belongs too. Should be one of 'Others', 'Directory services', 'Cloud/OS', 'Integration', 'Cisco firewall', 'Authentication', 'Cisco dna', 'Design'",
        type = 'string'
    ),
    AttributeInfo(
        name = 'topic_name',
        description = "The topic where the document belongs to.",
        type = 'sstring'

    )
]



def get_context(class_name, query):

    db = Chroma(persist_directory='vector_databases/Licensing_db', 
                      embedding_function=EMBEDDINGS)
    
    print("Similar documents\n\n: ", db.similarity_search(query))


    print(f'got db from {db._persist_directory}')

    document_content_description = f"ISE product {class_name} documents"

    # document_content_description = "ISE product category documents"

    # llm = VertexAI(temperature=0)

    # query = "Give me discussions where a customer is confused in pxgrid licensing requirements"


    retriever = SelfQueryRetriever.from_llm(
        llm=LLM, 
        vectorstore=db,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )


    context = db.similarity_search(query,k=4)


    print("Query: \n", query)
    print()
    print("Context: \n", context)



    # print("Context: ", context)
    # print(context)

    # print(f'got {len(related_docs)} documents for context.\n\n{related_docs}')

    # context = '\n\n'.join([page.page_content for page in related_docs])

    return context



# examples = [
#     InputOutputTextPair(
#         input_text = 'What solutions were provided to the user?',
#         output_text = """The user was provided with the following solutions:\n\n
#         1. The user was advised to stay on 3.1.0.518 (Patch 5) until the fix is released.\n\n
#         2. The user was advised to contact TAC for further assistance.
#         """
#     ),

# InputOutputTextPair(
#      input_text ='Give me the whole conversation on the issue where user is unable to do accounting for device administration without a TACACS device admin license.',
#      output_text =  """
#          - The user is unable to do accounting for device administration without a TACACS device admin license.\n\n
#          - The user asks if there is a plan to support Device Management (TACACS) on an Extra Small VM (with no other features enabled).\n\n
#          - The support representative responds that the Extra Small VM is supported for PSN services only, this does include Device Admin.\n\n
#          - The user asks if there is another way to do accounting for device administration without a TACACS device admin license.\n\n
#          - The support representative responds that there is no other means today, unfortunately. They have this CSCuq59764 and will check with their teams to see if it can be prioritized soon.
#     """
# ),

# InputOutputTextPair(
#      input_text = 'Can you give me the whole conversation between the user and support representative where the User wants to integrate Cisco ISE 2.7 with Oracle Database Cloud but they are not sure if it is possible or how to do it.',
#      output_text =  """The following is the conversation between the user and support representative:\n\n
#     User: I am trying to integrate Cisco ISE 2.7 with Oracle Database Cloud, but I am not sure if it is possible or how to do it.\n\n
#     Support Representative: The only method ISE 2.7 would have to query an external Oracle database would be using ODBC. See this document for an example.\n\n
#     User: Thank you for the information. I will try using ODBC to integrate ISE 2.7 with Oracle Database Cloud.\n\n
#     Support Representative: You're welcome. Let me know if you have any other questions.
#     """
# )
# ]




examples = [
    {
        'query': 'What solutions were provided to the user?',
        'answer': """The user was provided with the following solutions:\n\n
        1. The user was advised to stay on 3.1.0.518 (Patch 5) until the fix is released.\n\n
        2. The user was advised to contact TAC for further assistance.
        """
    },

{
    'query': 'Give me the whole conversation on the issue where user is unable to do accounting for device administration without a TACACS device admin license.',
    'answer': """
         - The user is unable to do accounting for device administration without a TACACS device admin license.\n\n
         - The user asks if there is a plan to support Device Management (TACACS) on an Extra Small VM (with no other features enabled).\n\n
         - The support representative responds that the Extra Small VM is supported for PSN services only, this does include Device Admin.\n\n
         - The user asks if there is another way to do accounting for device administration without a TACACS device admin license.\n\n
         - The support representative responds that there is no other means today, unfortunately. They have this CSCuq59764 and will check with their teams to see if it can be prioritized soon.
    """
},

{
    'query': 'Can you give me the whole conversation between the user and support representative where the User wants to integrate Cisco ISE 2.7 with Oracle Database Cloud but they are not sure if it is possible or how to do it.',
    'answer': """The following is the conversation between the user and support representative:\n\n
    User: I am trying to integrate Cisco ISE 2.7 with Oracle Database Cloud, but I am not sure if it is possible or how to do it.\n\n
    Support Representative: The only method ISE 2.7 would have to query an external Oracle database would be using ODBC. See this document for an example.\n\n
    User: Thank you for the information. I will try using ODBC to integrate ISE 2.7 with Oracle Database Cloud.\n\n
    Support Representative: You're welcome. Let me know if you have any other questions.
    """
}
]



# }


# implement_db = get_db('implement')
# upgrade_db = get_db('upgrade')
# licensing_db = get_db("licensing")
# di_db = get_db('design_integration')






# implement_qa_chain = create_qa_chain('Implement')
# print('created Implement chain')
# upgrade_qa_chain = create_qa_chain('Upgrade')
# print('created Upgrade chain')
# licensing_qa_chain = create_qa_chain('Licensing')
# print('created Licensing chain')
# design_integration_qa_chain = create_qa_chain('Design/Integration')
# print('created Design/Integration chain')


if __name__ == '__main__':

    persist_directory = 'vbs/Licensing_db'

    embeddings = VertexAIEmbeddings()

    vectordb = Chroma(persist_directory=persist_directory, 
                      embedding_function=embeddings)
    
    document_content_description = "ISE product category documents"

    llm = VertexAI(temperature=0)

    retriever = SelfQueryRetriever.from_llm(
        llm=LLM, 
        vectorstore=vectordb,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )

    query = "Give me discussions where a  customer wants to smart license their ISE box which is not connected to the internet."

    
    print(retriever.get_relevant_documents(query))