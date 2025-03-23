#!/usr/bin/env python3.10.16
"""文档分割组件.

利用langchian组件编写文档分割相关工具函数.

Copyright 2025 Kai.
License(GPL)
Author: Kai
"""
from typing import Optional, Iterable, Literal, List
from langchain_core.documents import Document
from langchain_text_splitters import SpacyTextSplitter

class Splitter:
    """文档文本分割组件."""
    @staticmethod
    def split_docs(
        target: Document | Iterable[Document] | str,
        mode: Literal["text", "doc", "docs"] = "docs",
        separator: Optional[str] = None,
        pipeline: Optional[str] = None,
        max_length: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strip_whitespace: bool = True,
        ) -> List[Document] | List[str]:
        """分割文档.
        
        利用spacy分割文档, 文档列表或者文本.
        根据语义分割, 不支持特殊格式文件的分割, 如json.
        不同语言的分割方式不同, 需要根据语言选择不同的pipeline并下载对应的模型.
        例如:
        en_core_web_sm: 英语
        zh_core_web_sm: 中文

        Args:
            target (Document | Iterable[Document] | str): 需要分割的文档列表或文本.
            mode (Literal["text", "doc", "docs"]): 分割模式, 默认为"docs".
            separator (Optional[str]): 分割符, 默认为\\n\\n.
            pipeline (Optional[str]): 分割方式, 默认使用en_core_web_sm.
            max_length (Optional[int]): 分割前的文档最大长度, 默认1_000_000.
            chunk_size (Optional[int]): 每个块的最大长度, 默认4000.
            chunk_overlap (Optional[int]): 块之间的重叠长度, 默认200.
            strip_whitespace (bool): 是否去除空白字符, 默认为True.

        Returns:
            list[Document]: 分割后的文档列表.
            list[str]: 分割后的文本列表.
        """
        kargs = {}
        if separator:
            kargs['separator'] = separator
        if pipeline:
            kargs['pipeline'] = pipeline
        if max_length:
            kargs['max_length'] = max_length
        if chunk_size:
            kargs['chunk_size'] = chunk_size
        if chunk_overlap:
            kargs['chunk_overlap'] = chunk_overlap
        splitter = SpacyTextSplitter(
            strip_whitespace=strip_whitespace,
            **kargs
        )
        if mode == "text":
            return splitter.split_text(target)
        elif mode == "doc":
            return splitter.split_documents([target])
        elif mode == "docs":
            return splitter.split_documents(target)
