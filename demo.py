#!/usr/bin/env python3.10.16
"""代码使用样例.

利用组件编写代码使用样例.

Copyright 2025 Kai.
License(GPL)
Author: Kai
""" 
from src.components import (
    SpecificFileLoader,
    Splitter,
    ChromaCollection,
    EmbeddingFromHF
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# 数据存入流程
# 加载pdf文件
files = SpecificFileLoader.pdf_load_dir("./files/test_docs")
# 分割文档
files = Splitter.split_docs(
    files,
    pipeline="zh_core_web_sm",
    chunk_size=900,
    chunk_overlap=100
)
# 下载嵌入模型
# EmbeddingFromHF.download_model("BAAI/bge-m3", "./files/models/bge-m3")
# 加载嵌入模型
embedding = EmbeddingFromHF.load_model("./files/models/bge-m3", "cuda:0", True)
# 加载或创建向量数据库集合
db = ChromaCollection("chroma_langchain_db", "./files/database/chroma_langchain_db", embedding)
# 清空集合
db.coll_clear()
# 添加文档
db.add_documents(files)


# 构建查询流程
# 向量数据库
vectorstore = db.vector_store
# 检索器
retriever = vectorstore.as_retriever()
# 构建prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
# 构建检索和prompt的并行流程
retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
prompt = ChatPromptTemplate.from_template(template)
# 构建模型
model = ChatOpenAI(model="deepseek-chat")
# 构建输出解析器
output_parser = StrOutputParser()
# 构建chain
chain = retrieval | prompt | model | output_parser
# 执行chain
ans = chain.invoke("介绍一下Transformer的基本架构")
print(ans)