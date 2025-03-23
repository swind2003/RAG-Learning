#!/usr/bin/env python3.10.16
"""向量数据库组件.

利用langchian组件编写向量数据库相关工具函数.

Copyright 2025 Kai.
License(GPL)
Author: Kai
""" 
from typing import Optional, List, Union
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class ChromaCollection:
    """Chroma向量数据库集合类.
    
    Attributes:
        name (str): 集合名称.
        embedding_model (HuggingFaceEmbeddings): 嵌入模型.
        local_directory (str): 本地目录.
        vector_store (Chroma): 向量数据库.
    """
    def __init__(self,
        collection: str,
        directory: Optional[str]=None,
        embedding: Optional[HuggingFaceEmbeddings]=None
        ):
        """初始化Chroma向量数据库集合.
        
        Args:
            collection (str): 集合名称.
            embedding (HuggingFaceEmbeddings): 嵌入模型.
            directory (str): 本地目录.
        """
        if not collection:
            raise ValueError("collection 不能为空")
        
        self.name = collection
        self.embedding_model = embedding
        self.local_directory = directory
        self.coll_create()
        
    def add_documents(self,
        documents: Union[Document, List[Document]]
        ):
        """向向量数据库添加文档.
        
        Args:
            documents (Union[Document, List[Document]]): 要添加的文档或文档列表
        """
        if isinstance(documents, Document):
            documents = [documents]
        
        self.vector_store.add_documents(documents)
        
    def delete_documents(self,
        ids: Union[str, List[str]]
        ):
        """从向量数据库删除文档.
        
        Args:
            ids (Union[str, List[str]]): 要删除的文档ID或ID列表
        """
        if isinstance(ids, str):
            ids = [ids]
            
        self.vector_store.delete(ids)
        
    def update_documents(self,
        ids: Union[str, List[str]],
        documents: Union[Document, List[Document]]
        ):
        """更新向量数据库中的文档.
        
        先删除原有文档，再添加新文档.
        
        Args:
            ids (Union[str, List[str]]): 要更新的文档ID或ID列表
            documents (Union[Document, List[Document]]): 新的文档内容
        """
        if isinstance(ids, str):
            ids = [ids]
        
        self.vector_store.update_documents(ids, documents)
        
    def search(self,
        query: str,
        k: int = 4
        ) -> List[Document]:
        """根据查询字符串搜索最相似的文档.
        
        Args:
            query (str): 查询字符串
            k (int): 返回的最相似文档数量, 默认为4
            
        Returns:
            List[Document]: 最相似的文档列表
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def search_with_score(self,
        query: str,
        k: int = 4
        ) -> List[tuple[Document, float]]:
        """根据查询字符串搜索最相似的文档并返回相似度分数.
        
        Args:
            query (str): 查询字符串
            k (int): 返回的最相似文档数量,默认为4
            
        Returns:
            List[tuple[Document, float]]: 最相似的文档及其相似度分数的列表
        """
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def search_by_vector(self,
        embedding: List[float],
        k: int = 4
        ) -> List[Document]:
        """根据向量搜索最相似的文档.
        
        Args:
            embedding (List[float]): 查询向量
            k (int): 返回的最相似文档数量, 默认为4
            
        Returns:
            List[Document]: 最相似的文档列表
        """
        return self.vector_store.similarity_search_by_vector(embedding, k=k)
 
    def coll_get(self
        ) -> Chroma:
        """获取向量数据库集合."""
        return self.vector_store
    
    def coll_list(self
        ) -> List[Document]:
        """获取向量数据库中的所有文档.
        
        Returns:
            List[Document]: 所有文档列表
        """
        return self.vector_store.get()
    
    def coll_create(self):
        """创建向量数据库集合."""
        self.vector_store = Chroma(
            collection_name=self.name,
            embedding_function=self.embedding_model,
            persist_directory=self.local_directory
        )
    
    def coll_clear(self):
        """清空向量数据库集合."""
        self.vector_store.reset_collection()
    
    def coll_destroy(self):
        """销毁向量数据库集合."""
        self.vector_store.delete_collection()
