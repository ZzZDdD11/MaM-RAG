# app/api/routers/ingest.py
import os
import shutil
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.core.graph_extract import extract_and_store_graph
# 1. 导入文档解析工具
from docling.document_converter import DocumentConverter

# 2. 导入 LangChain 的切分工具
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 3. 导入我们刚才写的向量库单例
from app.core.vector import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

class IngestResponse(BaseModel):
    status: str
    message: str
    filename: str

def _process_and_insert(file_path: str, original_filename: str):
    """
    后台任务：解析 -> 切分 -> 存入 Milvus
    """
    try:
        # --- 第一步：Docling 解析 ---
        logger.info(f"📄 [1/3] 正在解析文件: {original_filename}")
        converter = DocumentConverter()
        result = converter.convert(file_path)
        # 导出为 Markdown，保留了标题层级结构
        full_text = result.document.export_to_markdown()
        
        if not full_text.strip():
            logger.warning(f"⚠️ 文件 {original_filename} 解析为空，跳过。")
            return

        # --- 第二步：智能切分 (Chunking) ---
        logger.info(f"🔪 [2/3] 正在切分文档...")
        
        # 使用"递归字符切分器"，这是目前最通用的策略
        # 它会优先在段落(\n\n)、句子(。)之间切分，尽量不切断语义
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,       # 每个块大约 600 字符
            chunk_overlap=100,    # 重叠 100 字符，防止上下文丢失
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        # 将文本切分成 Document 对象列表
        # metadata 非常重要！以后我们可以根据 source 筛选特定的文件
        chunks = text_splitter.create_documents(
            [full_text], 
            metadatas=[{"source": original_filename}]
        )
        
        logger.info(f"📦 切分完成，共生成 {len(chunks)} 个文本块。")

        # --- 第三步：存入 Milvus ---
        logger.info(f"💾 [3/3] 正在写入 Milvus 数据库...")
        
        vector_store = get_vector_store()
        #这一步会自动调用 HuggingFace 模型把文本变成向量，然后存入 Milvus
        vector_store.add_documents(chunks)
        logger.info("向量入库成功")

        logger.info(f"⛏️ [4/4] 正在进行图谱抽取与存储...")
        extract_and_store_graph(chunks)
        
        logger.info(f"🎉 文件 {original_filename} 全部处理完成！")

    except Exception as e:
        logger.error(f"❌ 入库失败 {original_filename}: {e}", exc_info=True)
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/ingest/file", response_model=IngestResponse, summary="上传文件到 Milvus")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    接收 PDF/Docx/MD 文件，后台异步处理并存入向量库。
    """
    # 格式检查
    allowed_exts = ('.pdf', '.docx', '.md', '.txt')
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail=f"仅支持: {allowed_exts}")

    try:
        # 保存上传文件到临时目录
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 启动后台任务 (不阻塞接口返回)
        background_tasks.add_task(_process_and_insert, tmp_path, file.filename)

        return IngestResponse(
            status="accepted",
            message="文件已接收，正在后台解析并入库...",
            filename=file.filename
        )

    except Exception as e:
        logger.error(f"上传接口报错: {e}")
        raise HTTPException(status_code=500, detail=str(e))