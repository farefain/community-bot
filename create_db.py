import pandas as pd 
import re 
from bs4 import BeautifulSoup
from langchain.docstore.document import Document 
from langchain.embeddings import VertexAIEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os 
from langchain.vectorstores import DocArrayInMemorySearch

def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()   
    clean_doc = re.sub(r'\xa0', '', text)
    return clean_doc



class_names = ['Upgrade', 'Licensing', 'Implement', 'Design/Integration']

if os.path.exists(f'vector_databases'):
    print("Database already exists")
else:
    os.mkdir(f'vector_databases')



# create a function that takes in a class name and creates a database for that class
def create_db(class_names):
    df = pd.read_excel('dataset/tag_clusters.xlsx')
    df = df[df['analysis_window'] == 'annual']
    df = df[(df['class'] == class_names)]
    df['text'] = df['text'].apply(remove_html_tags)


for class_name in class_names:

    print(f'Processing -- {class_name}')

    df = pd.read_excel('dataset/tag_clusters.xlsx')
    df = df[df['analysis_window'] == 'annual']
    print(df['class'].value_counts())

    df = df[(df['class'] == class_name)]
    print(df['class'].value_counts())
    print('analysis window: ', df['analysis_window'].value_counts())


    df['text'] = df['text'].apply(remove_html_tags)
    print("cleaned documents ><")

    docs = df['text'].tolist()
    post_url = df['post_url'].tolist()
    post_tag = df['tag'].tolist()
    post_topic = df['topic_name'].tolist()
    post_summary = df['summary_text'].tolist()
    conversation_id = df['conversation_id'].tolist()

    # convert nan values to ''
    docs = [doc if isinstance(doc, str) else '' for doc in docs]
    post_summary = [summary if isinstance(summary, str) else '' for summary in post_summary]

    try:

        doc_augmented = [post_summary + " " + doc for post_summary, doc in zip(post_summary, docs)]

    except TypeError:
        print("TypeError encountered")
        print("post_summary: ", post_summary)
        print("doc: ", docs)

    docs = [Document(page_content=doc, 
                     metadata={'url': post_url[i], 
                               'tag': post_tag[i], 
                               'topic_name': post_topic[i], 
                               'summary': post_summary[i], 
                               'conversation_id': conversation_id[i]}) for i, doc in enumerate(doc_augmented)]
    
    print("Converted text into document objects <>")

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 20,
            separators=['||||']
    )

    docs = text_splitter.split_documents(docs)
    print("Documents split <<>>")

    embeddings = VertexAIEmbeddings()

    if class_name == 'Design/Integration':
        db_name = 'Design_Integration'
    else:
        db_name = class_name

    
   

    print("Persisting database")
    db = DocArrayInMemorySearch.from_documents(docs, embeddings, persist_directory=f'vectors/{db_name}_db')
    db.persist()

    print("database directory: ", db._client_settings.persist_directory)


