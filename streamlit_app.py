import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from pathlib import Path
import json
from langchain.text_splitter import CharacterTextSplitter
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Qdrantの設定
COLLECTION_NAME = "gci_2024_winter"

# タイトルと説明を表示
st.title("💬 RAG機能付きチャットボット")
st.write(
    "このチャットボットは、OpenAIのGPT-3.5モデルとドキュメント検索を組み合わせています。 "
    "このアプリを使うには、OpenAI APIキーを入力してください。APIキーは[こちら](https://platform.openai.com/account/api-keys)で取得できます。"
)

# ユーザーからOpenAI APIキーを取得
openai_api_key = st.text_input("OpenAI APIキー", type="password")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(":memory:")  # インメモリモードで初期化

def load_qdrant(client):
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        st.write('新しいコレクションを作成しました。')
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    )

def build_vector_store(texts, client):
    qdrant = load_qdrant(client)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=250,
        chunk_overlap=0,
    )
    documents = text_splitter.create_documents(texts)
    qdrant.add_documents(documents)
    st.write('ベクトルストアにドキュメントを追加しました。')
    return qdrant

if not openai_api_key:
    st.info("APIキーを入力してください。", icon="🗝️")
else:
    # OpenAIクライアントの設定
    client = OpenAI(api_key=openai_api_key)

    # データ読み込み
    workdir = Path('.')  # 実際の作業ディレクトリに置き換えてください
    datapath = workdir / "gci.json"
    
    with datapath.open('r', encoding='utf-8') as file:
        data = json.load(file)

    # JSONデータの内容をテキストに変換
    texts = [json.dumps(d, ensure_ascii=False) for d in data]
    st.write(f"読み込んだテキスト数: {len(texts)}")

    # ドキュメントのサイズ（トークン数）を確認
    encoding = tiktoken.encoding_for_model('text-embedding-3-small')
    len_docs = [len(encoding.encode(text)) for text in texts]

    avg_doc_length = round(sum(len_docs) / len(len_docs), 1)
    max_doc_length = max(len_docs)

    st.write(f"ドキュメントの平均トークン数: {avg_doc_length}")
    st.write(f"ドキュメントの最大トークン数: {max_doc_length}")

    if max_doc_length > 5000:
        st.warning("> トークン数が5,000を超えるドキュメントが存在するので、トークン数を抑える何らかの対処が必要", icon="⚠️")

    # ベクトルストアの構築
    build_vector_store(texts)

    # レトリーバーの設定
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(search_kwargs={"k": 3})

    # プロンプトテンプレートの定義
    template = """
    あなたはGCI 2024 Winterの講師です。
    
    次の参考情報を参考に、ユーザーの質問に答えてください。
    
    # 制約
    事実とアドバイスを分けて教えてください。
    
    # 参考情報
    {context}
    
    # ユーザーの質問
    {question}
    
    # 出力
    事実：
    アドバイス：
    回答文書：
    """
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        template_format="f-string"
    )
    
    # LLMの設定
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=openai_api_key, temperature=0.0)
    
    # RetrievalQAチェーンの作成
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True,
    )
    
    # セッション状態にチャットメッセージを保存
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 過去のチャットメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ユーザーからの質問入力
    if prompt := st.chat_input("質問をどうぞ"):
        # ユーザーの質問を保存して表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # RAGモデルを使って回答を生成
        response = qa.invoke(prompt)
        
        # アシスタントの回答を表示して保存
        answer = response['result']
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)