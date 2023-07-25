import streamlit as st
import pandas as pd 
from chains import (EMBEDDINGS, 
                    # implement_qa_chain, 
                    # upgrade_qa_chain, 
                    # licensing_qa_chain, 
                    # design_integration_qa_chain, 
                    LLM,
                    get_context,
                    examples,
                  
                    metadata_field_info
)
# from langchain.chat_models import ChatVertexAI
from langchain.chat_models.vertexai import ChatVertexAI
import ptvsd
from langchain import FewShotPromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# from langchain.prompts import (
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
#     MessagesPlaceholder

# )

from langchain.prompts import PromptTemplate
from streamlit_chat import message
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.self_query.base import SelfQueryRetriever

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import random 

# ptvsd.enable_attach(address=('localhost', 5678))
# ptvsd.wait_for_attach()


# col1, col2 = st.columns([3, 1])

# with col1:


# Page title
#   st.set_page_config(page_title='🦜🔗 :ear: Context AI Communities')
st.title('👂 Context AI - Communities')
st.markdown("""---""")



query_class = st.sidebar.selectbox('Classes', ['Upgrade', 'Licensing', 'Design_Integration', 'Implement'])
db = Chroma(persist_directory=f'vector_databases/{query_class}_db', 
                      embedding_function=EMBEDDINGS)
    
document_product_tag = st.sidebar.selectbox('Tags', ['Others', 'Directory services', "Cloud/OS", 'Integration', 'Cisco firewall', 'Authentication', 'Cisco dna', 'Design'])
st.sidebar.markdown("---")
tag_influence=st.sidebar.checkbox('Chat Tag Filter', value=True)
st.sidebar.markdown("---")


with st.expander('click to see clusters'):
    # read tag cluster document and filter for class-tag-annual combination



    # import top words df and filter for class-tag-annual combination 
    top_words_df = pd.read_excel('dataset/top_topics_aq.xlsx')
    top_words_df = top_words_df[(top_words_df['class'] == query_class) & (top_words_df['product_tag'] == document_product_tag) & (top_words_df['analysis_window'] == 'annual')]
    unique_clusters = top_words_df['topic_id'].unique().tolist()
    top_fig = go.Figure()
    for i in unique_clusters:
        # get top words, scores and topic id 
        top_words = top_words_df[top_words_df['topic_id'] == i]['word'].tolist()
        top_scores = top_words_df[top_words_df['topic_id'] == i]['score'].tolist()
        topic_id = 'Topic ' + str(i)
        top_fig.add_trace(go.Scatter(x=top_words, 
                                 y=top_scores, 
                                 name=topic_id,
                                 marker=dict(
                                     size=top_scores,
                                    sizemode='diameter',
                                    sizeref=0.1,
                                    sizemin=5,
                                    color=random.choice(px.colors.qualitative.Plotly)
                                 )
                            )
                    )

    if len(unique_clusters) >= 2:
        top_fig.update_layout(shapes=[
            dict(
                type= 'line',
                yref='paper',
                xref='x',
                x0=len(top_words_df[top_words_df['topic_id'] == 0]['word'].tolist())-0.5,
                y0=0,
                x1=len(top_words_df[top_words_df['topic_id'] == 0]['word'].tolist())-0.5,
                y1=1,
                line=dict(color='black', width=2)
            )
        ])

    # add labels and title 
    top_fig.update_layout(
        title='Top 10 Words',
        xaxis_title='Words',
        yaxis_title='Weight',)

    


    df = pd.read_excel('dataset/tag_clusters.xlsx')
    df = df[df['analysis_window'] == 'annual']
    df_scatter = df[(df['class'] == query_class) & (df['tag'] == document_product_tag)]
 

    fig = px.scatter(
    df_scatter,
    x="post_x_coordinate", 
    y="post_y_coordinate",

    # size="pop",
    color="topic_id",
    hover_name="summary_text",
    # log_x=True,
    size_max=60,
    )

    tab1, tab2 = st.tabs(["Cluster Visualizations", "Top Words"])
    with tab1:
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with tab2:
        # Use the native Plotly theme.
        st.plotly_chart(top_fig, theme="streamlit", use_container_width=True)



with st.sidebar.expander('click to see cluster tag summary'):
    df = pd.read_excel('dataset/cluster_summaries_aq.xlsx')
    class_tag_summary = df[(df['class'] == query_class) & (df['tag'] == document_product_tag) & (df['analysis_window'] == 'annual')]
    class_tag_summary = class_tag_summary[['topic_id', 'summary']]
    
    # iterate over the rows and write the summary
    for doc in class_tag_summary.iterrows():
        st.write(doc[1]['topic_id'])
        st.write(doc[1]['summary'])
        st.markdown("""---""")

st.sidebar.subheader('Context')


# summary_df = pd.read_excel('dataset/cluster_summaries_aq.xlsx')
# class_tag_summary = summary_df[(summary_df['class'] == query_class) & (summary_df['tag'] == summary_tag) & (summary_df['analysis_window'] == 'annual')]

# col_nums = len(class_tag_summary) 

# with st.expander(f"Click to show summary for {query_class}"):
#     col1, col2 = st.columns(2)

#     with col1:
#         st.write('Summary of topic 1')

#     with col2:
#         st.write('Summary of topic 2')


# st.markdown("""---""")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# if 'context' not in st.session_state:
#     st.session_state['context'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True, memory_key='chat_history', input_key='query',)



# query_class = st.selectbox('Select', ['Upgrade', 'Licensing', 'Design_Integration', 'Implement'])
# query_class = st.sidebar.selectbox('Classes', ['Upgrade', 'Licensing', 'Design_Integration', 'Implement'])


# st.table(df['summary'].iloc[0:10])

# # Query text
# query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not(query_class))

# sysprompt = """From the history and context provided below, Answer the question as truthfully as possible. 
#     and if the answer is not contained within the text below, say 'I don't know' or ask me to refine my query. 

# Current conversation:
# """


# system_msg_template = SystemMessagePromptTemplate.from_template(template=sysprompt)



# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")


template = """You are an awesome question answering assistant. 
              Answer the QUESTION based on CURRENT CONVERSATION and CONTEXT  given below. 
              Answer the question as truthfully as possible. Provide a link, if provided within a conversation.
              If the answer is not contained within the text below, say 'I don't know' or ask to refine my query. 

              Here are some examples you should follow in answering questions:

             
CURRENT CONVERSATION:
```
{chat_history}
````

CONTEXT:
```
{context}
```

QUESTION:
```
{query}
```

ANSWER: 
"""



prompt = PromptTemplate(
    template=template, 
    input_variables=['chat_history', 'context', 'query']
)

conversation = load_qa_chain(memory=st.session_state.buffer_memory, 
                                 prompt=prompt, 
                                 llm=LLM, 
                                 chain_type='stuff',
                                 verbose=True)

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()




def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string



with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("responding..."):
           
            
            if tag_influence:
                context = db.similarity_search(query, 
                                               k=4, 
                                               return_documents=True, 
                                               filter={'tag': document_product_tag})
            else:
                context = db.similarity_search(query, 
                                               k=4, 
                                               return_documents=True)
                

            with st.sidebar.expander(f"Context"):

                
                prev_conv_id = ''
                for document in context:

                    conv_id = document.metadata['conversation_id']
                    if conv_id == prev_conv_id:
                        continue
                    st.sidebar.write(document.metadata['summary'])
                    st.sidebar.write(document.metadata['tag'])
                    st.sidebar.write(document.metadata['url'])
                    st.sidebar.markdown("""---""")
                    prev_conv_id = conv_id
                    

            # context = '\n\n'.join([page.page_content for page in context])

            print("Context Documents: \n", context)

            query_dict ={
                'input_documents': context,
                'query': query

            }

            # CONTENT = template.format(chat_history=st.session_state.buffer_memory,
            #                     context=context,
            #                     query=query)
            
            response = conversation.run(query_dict)

            print(response)

         
          
            # response = conversation(query_dict, return_only_outputs=True)
            # response = qa({"question": query})
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
    
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')



# with col2:
#   st.title('Context')
#   st.write(st.session_state.context)





# # Form input and query
# # result = []
# # with st.form('myform', clear_on_submit=True):
# #     # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
# #     submitted = st.form_submit_button('Submit', disabled=not(query_text and query_class))
# #     if submitted:        
# #         with st.spinner('Calculating...'):
            
# #             response = get_response(query_class, query_text)
# #             result.append(response)

# # if len(result):
# #     st.info(response)