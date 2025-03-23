#!/usr/bin/env python3.10.16
"""嵌入模型组件.

利用langchian组件编写嵌入模型相关工具函数.

Copyright 2025 Kai.
License(GPL)
Author: Kai
""" 
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download

class EmbeddingFromHF:
    """HuggingFace来源的嵌入模型."""

    @staticmethod
    def download_model(
        repo_id: str,
        local_dir: str,
        use_symlinks: bool=False
        ) -> None:
        """下载模型.
        
        Args:
            repo_id (str): 模型ID.
            local_dir (str): 下载到的本地目录.
        """
        if not local_dir or not repo_id:
            raise ValueError("local_dir 和 repo_id 不能为空")
        
        snapshot_download(
            repo_id=repo_id, 
            local_dir=local_dir,
            # 不使用符号链接直接复制
            local_dir_use_symlinks=use_symlinks
        )

    @staticmethod
    def load_model(
        local_dir: str,
        device: str="cpu",
        normalize_embeddings: bool=False
        ) -> HuggingFaceEmbeddings:
        """加载模型.
        
        Args:
            local_dir (str): 本地目录.
        
        Returns:
            embeddings (HuggingFaceEmbeddings): 嵌入模型.
        """
        if not local_dir:
            raise ValueError("local_dir 不能为空")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=local_dir,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
        )
        return embeddings

