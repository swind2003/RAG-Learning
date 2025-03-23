#!/usr/bin/env python3.10.16
"""文档加载组件.

利用langchian组件编写文档加载相关工具函数.

Copyright 2025 Kai.
License(GPL)
Author: Kai
"""
import os
from typing import Literal, Optional, Callable, List, Dict, Any
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import (
    TextLoader, 
    CSVLoader, 
    JSONLoader, 
    PyPDFLoader, 
    PDFMinerLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.document_loaders.parsers.images import LLMImageBlobParser
from langchain_openai import ChatOpenAI

class MultiFileLoader:
    """多文件加载器.
    
    支持多种单个文件加载方法:
    - .txt
    - .csv
    - .json
    - .pdf
    - .docx
    - .xls
    - .xlsx
    - .md
    """
    @staticmethod
    def load_text_file(
        file_path: str,
        encoding: str = "utf-8"
        ) -> List[Document]:
        """加载文本文件.
        
        Args:
            file_path (str): 文件路径.
            encoding (str): 文件编码.
            
        Returns:
            List[Document]: 文档列表.
        """
        loader = TextLoader(file_path, encoding=encoding)
        return loader.load()
    
    @staticmethod
    def load_csv_file(
        file_path: str, 
        csv_args: Dict[str, Any] = None,
        column_names: List[str] = None,
        source_column: str = None
        ) -> List[Document]:
        """加载CSV文件.
        
        Args:
            file_path (str): 文件路径.
            csv_args (Dict[str, Any]): CSV读取参数.
            column_names (List[str]): 列名列表.
            source_column (str): 源列名.
            
        Returns:
            List[Document]: 文档列表.
        """
        if csv_args is None:
            csv_args = {}
        loader = CSVLoader(
            file_path, 
            csv_args=csv_args,
            column_names=column_names,
            source_column=source_column
        )
        return loader.load()
    
    @staticmethod
    def load_json_file(
        file_path: str,
        jq_schema: str = ".[]",
        text_content: bool = False
        ) -> List[Document]:
        """加载JSON文件.
        
        Args:
            file_path (str): 文件路径.
            jq_schema (str): JQ查询模式.
            text_content (bool): 是否以文本形式返回内容.
            
        Returns:
            List[Document]: 文档列表.
        """
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=jq_schema,
            text_content=text_content
        )
        return loader.load()
    
    @staticmethod
    def load_pdf_file(
        file_path: str,
        password: Optional[str] = None
        ) -> List[Document]:
        """加载PDF文件.
        
        Args:
            file_path (str): 文件路径.
            password (Optional[str]): PDF密码.
            
        Returns:
            List[Document]: 文档列表.
        """
        loader = PyPDFLoader(file_path=file_path, password=password)
        return loader.load()

    @staticmethod
    def load_docx_file(
        file_path: str,
        mode: str = "single"
        ) -> List[Document]:
        """加载Word文档.
        
        Args:
            file_path (str): 文件路径.
            mode (str): 加载模式, 可选值: "single"、"elements".
            
        Returns:
            List[Document]: 文档列表.
        """
        loader = UnstructuredWordDocumentLoader(file_path=file_path, mode=mode)
        return loader.load()
    
    @staticmethod
    def load_excel_file(
        file_path: str,
        mode: str = "single"
        ) -> List[Document]:
        """加载Excel文件.
        
        Args:
            file_path (str): 文件路径.
            mode (str): 加载模式, 可选值: "single"、"elements".
            
        Returns:
            List[Document]: 文档列表.
        """
        loader = UnstructuredExcelLoader(file_path=file_path, mode=mode)
        return loader.load()
    
    @staticmethod
    def load_markdown_file(
        file_path: str,
        mode: str = "single"
        ) -> List[Document]:
        """加载Markdown文件.
        
        Args:
            file_path (str): 文件路径.
            mode (str): 加载模式, 可选值: "single"、"elements".
            
        Returns:
            List[Document]: 文档列表.
        """
        loader = UnstructuredMarkdownLoader(file_path=file_path, mode=mode)
        return loader.load()

class BlindFileLoader:
    """盲文件加载器.
    
    根据文件扩展名自动选择合适的加载器.
    """
    @staticmethod
    def load_any_file(
        file: Optional[str | list[str]] = None,
        key: str=os.getenv("UNSTRUCTURED_API_KEY"),
        ) -> List[Document]:
        """加载任意文件.
        
        json类文件只接受ndjson格式.
        
        Args:
            file (Optional[str | list[str]]): 文件路径.
            key (str): 结构化API密钥.
        
        Returns:
            List[Document]: 文档列表.
        """
        try:
            loader=UnstructuredLoader(
                file_path=file,
                partition_via_api=True,
                api_key=key
            )
            return loader.load()
        except Exception as e:
            print(f"加载文件 {file} 时出错: {str(e)}")
            return []

    @staticmethod
    def load_every_file(
        dir_path: str,
        key: str=os.getenv("UNSTRUCTURED_API_KEY"),
        ) -> List[Document]:
        """利用load_any_file读取文件夹下所有文件.
        
        Args:
            dir_path (str): 文件夹路径.
            key (str): 结构化API密钥.
        
        Returns:
            List[Document]: 文档列表.
        """ 
        files = os.listdir(dir_path)
        docs = []
        for file in files:
            file_path = os.path.join(dir_path, file)
            try:
                temp = BlindFileLoader.load_any_file(file_path, key=key)
                if temp:
                    docs.append(temp)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
        return docs

    @staticmethod
    def load_file(
        file_path: str,
        encoding: str = "utf-8", 
        mode: str = "single",
        password: Optional[str] = None,
        ) -> List[Document]:
        """加载任意类型的单个文件，根据文件扩展名自动选择合适的加载器.
        
        无法读取.doc文件.
        
        Args:
            file_path (str): 文件路径.
            encoding (str): 文本文件编码.
            mode (str): 非结构化加载器模式, 可选值: "single"、"elements".
            password (Optional[str]): PDF密码.
        
        Returns:
            List[Document]: 文档列表.
        """
        ext = os.path.splitext(file_path.lower())[1]
        
        try:
            if ext == '.txt':
                return MultiFileLoader.load_text_file(file_path, encoding=encoding)
            elif ext == '.csv':
                return MultiFileLoader.load_csv_file(file_path)
            elif ext == '.json':
                return MultiFileLoader.load_json_file(file_path)
            elif ext == '.pdf':
                return MultiFileLoader.load_pdf_file(file_path, password=password)
            elif ext == '.docx':
                return MultiFileLoader.load_docx_file(file_path, mode=mode)
            elif ext in ['.xlsx', '.xls']:
                return MultiFileLoader.load_excel_file(file_path, mode=mode)
            elif ext in ['.md', '.markdown']:
                return MultiFileLoader.load_markdown_file(file_path, mode=mode)
            else:
                # 对于未知类型，报错
                raise ValueError(f"未知文件类型: {ext}")
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return []

    @staticmethod
    def load_dir(
        dir_path: str,
        encoding: str = "utf-8",
        mode: str = "single",
        password: Optional[str] = None
        ) -> List[Document]:
        """加载目录中的所有文件.
        
        Args:
            dir_path (str): 目录路径.
            encoding (str): 文本文件编码.
            mode (str): 非结构化加载器模式, 可选值: "single"、"elements".
            password (Optional[str]): PDF密码.
        
        Returns:
            List[Document]: 文档列表.
        """
        files = os.listdir(dir_path)
        docs = []
        for file in files:
            file_path = os.path.join(dir_path, file)
            try:
                temp = BlindFileLoader.load_file(
                    file_path,
                    encoding=encoding,
                    mode=mode,
                    password=password
                )
                if temp:
                    docs.append(temp)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
        return docs

class SpecificFileLoader:
    """特定文件加载器.
    
    用更好的方法加载特定文件.
    """
    @staticmethod
    def pdf_load_file(
        file_path: str,
        mode: Literal["default", "lazy", "async"] = "default",
        from_web: Optional[dict] = None,
        password: Optional[str] = None,
        load_mode: Literal["single", "page"] = "single",
        single_delimiter: Optional[str] = None,
        include_img: bool = False,
        img_model: Optional[str] = None,
        img_format: Optional[Literal["text", "markdown-img", "html-img"]] = None,
        ) -> Optional[List[Document] | PDFMinerLoader]:
        """加载PDF文件.
        
        利用PDFMinerLoader加载PDF文件.
        由于读取图片需要调用ChatOpenAI, 所以读取图片时需要在环境变量中设置OPENAI_API_KEY.
        
        Args:
            file_path (str): 文件路径.
            from_web (Optional[dict]): 从网页加载PDF文件.
            password (Optional[str]): PDF密码.
            load_mode (Literal["single", "page"]): 加载模式.
            single_delimiter (Optional[str]): 单个文档分页分隔符.
            include_img (bool): 是否包含图片.
            img_model (Optional[str]): 图片模型.
            img_format (Optional[Literal["text", "markdown-img", "html-img"]]): 图片格式.
        
        Returns:
            List[Document]: 文档列表.
            PDFMinerLoader: 加载器.
            None: 加载失败.
        """
        # 检查文件类型是否为PDF
        _, ext = os.path.splitext(file_path.lower())
        if ext != '.pdf':
            # 如果不是PDF文件, 则返回None
            return None
        # 位置参数
        kargs = {}
        if single_delimiter and load_mode == "single":
            kargs["pages_delimiter"] = single_delimiter
        if include_img:
            if img_model:
                kargs["images_parser"] = LLMImageBlobParser(
                    model=ChatOpenAI(model=img_model)
                )
            if img_format:
                kargs["images_inner_format"] = img_format
        # 创建加载器
        loader = PDFMinerLoader(
            file_path=file_path,
            headers=from_web,
            password=password,
            mode=load_mode,
            extract_images=include_img,
            **kargs
        )
        # 根据模式选择加载方式
        match mode:
            case "default":
                return loader.load()
            case "lazy":
                return loader.lazy_load()
            case "async":
                return loader.aload()
            case _:
                raise ValueError(f"无效的加载模式: {mode}")
    
    @staticmethod
    def pdf_load_dir(
        path: str,
        mode: Literal["single", "page"] = "single",
        single_delimiter: Optional[str] = None,
        img_included: bool = False,
        img_model: Optional[str] = None,
        img_format: Optional[Literal["text", "markdown-img", "html-img"]] = None
        ) -> List[Document]:
        """加载目录中的所有PDF文件.
        
        利用pdf_load_file加载目录中的所有PDF文件.
        只能是本地文件夹, 只能是未加密文件.
        会一次性将目录中的所有PDF文件加载到内存中.
        由于调用了pdf_load_file, 所以读取图片时需要在环境变量中设置OPENAI_API_KEY.
        
        Args:
            path (str): 目录路径.
            mode (Literal["single", "page"]): 加载模式.
            single_delimiter (Optional[str]): 单个文档分页分隔符.
            img_included (bool): 是否包含图片.
            img_model (Optional[str]): 图片模型.
            img_format (Optional[Literal["text", "markdown-img", "html-img"]]): 图片格式.
        
        Returns:
            List[Document]: 文档列表.
        """
        files = os.listdir(path)
        docs = []
        for file in files:
            file_path = os.path.join(path, file)
            # 检查是否为PDF文件
            _, ext = os.path.splitext(file.lower())
            if ext != '.pdf':
                continue
            try:
                temp = SpecificFileLoader.pdf_load_file(
                    file_path,
                    load_mode=mode,
                    single_delimiter=single_delimiter,
                    include_img=img_included,
                    img_model=img_model,
                    img_format=img_format
                )
                if temp:
                    docs.extend(temp)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
        return docs
    
    @staticmethod
    def json_load_file(
        file_path: str,
        mode: Literal["default", "lazy", "async"] = "default",
        jq_schema: Optional[str] = None,
        content_key: Optional[str] = None,
        content_parsable: Optional[bool] = None,
        text_content: bool = False,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        json_lines: bool = False,
        ) -> Optional[List[Document] | JSONLoader]:
        """加载JSON文件.
        
        Args:
            file_path (str): 文件路径.
            mode (Literal["default", "lazy", "async"]): 加载模式.
            jq_schema (Optional[str]): JQ查询模版.
            content_key (Optional[str]): 内容键.
                如果 jq_schema 结果是对象列表, 用于从每个对象中提取内容的键.
            content_parsable (Optional[bool]): content_key是否可解析.
                如果为 True, content_key 会被视为 jq 模式;
                如果为 False, content_key 被视为简单字符串键.
            text_content (bool): JSONLoader是否严格要求文档内容必须是字符串类型.
                如果为 True, 系统期望提取的内容是字符串类型;
                如果为 False, 系统允许提取非字符串类型的内容.
            metadata_func (Optional[Callable[[Dict, Dict], Dict]]): 元数据函数.
                接受 jq_schema 提取的 JSON 对象和默认元数据, 返回更新后的元数据字典.
            json_lines (bool): 是否为每行一个 JSON 对象的JSON Lines文件.
            
        Returns:
            List[Document]: 文档列表.
            JSONLoader: 加载器.
            None: 加载失败.
        """
        # 检查文件类型是否为JSON
        _, ext = os.path.splitext(file_path.lower())
        if ext != '.json':
            # 如果不是JSON文件, 则返回None
            return None
        # 位置参数
        kargs = {}
        if jq_schema:
            kargs["jq_schema"] = jq_schema
        else:
            kargs["jq_schema"] = ".[]"
        if content_key:
            kargs["content_key"] = content_key
            if content_parsable:
                kargs["is_content_key_jq_parsable"] = content_parsable
        if metadata_func:
            kargs["metadata_func"] = metadata_func
        # 创建加载器
        loader = JSONLoader(
            file_path=file_path,
            text_content=text_content,
            json_lines=json_lines,
            **kargs
        )
        # 根据模式选择加载方式
        match mode:
            case "default":
                return loader.load()
            case "lazy":
                return loader.lazy_load()
            case "async":
                return loader.aload()
            case _:
                raise ValueError(f"无效的加载模式: {mode}")
    
    @staticmethod
    def json_load_dir(
        path: str,
        jq_schema: Optional[str] = None,
        content_key: Optional[str] = None,
        content_parsable: Optional[bool] = None,
        content_string: bool = False,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        json_lines: bool = False,
        ) -> List[Document]:
        """加载目录中的所有JSON文件.
        
        利用json_load_file加载目录中的所有JSON文件.
        会一次性将目录中的所有JSON文件加载到内存中.
        
        Args:
            path (str): 目录路径.
            jq_schema (Optional[str]): JQ查询模版.
            content_key (Optional[str]): 内容键.
            content_parsable (Optional[bool]): content_key是否可解析.
            content_string (bool): 指示内容是否为字符串格式.
            metadata_func (Optional[Callable[[Dict, Dict], Dict]]): 元数据函数.
            json_lines (bool): 是否为每行一个 JSON 对象的JSON Lines文件.
        
        Returns:
            List[Document]: 文档列表.
        """
        files = os.listdir(path)
        docs = []
        for file in files:
            file_path = os.path.join(path, file)
            # 检查是否为JSON文件
            _, ext = os.path.splitext(file.lower())
            if ext != '.json':
                continue
            try:
                temp = SpecificFileLoader.json_load_file(
                    file_path,
                    jq_schema=jq_schema,
                    content_key=content_key,
                    content_parsable=content_parsable,
                    text_content=content_string,
                    metadata_func=metadata_func,
                    json_lines=json_lines
                )
                if temp:
                    docs.extend(temp)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
        return docs
