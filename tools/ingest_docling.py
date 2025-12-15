import argparse
import asyncio
import os
import logging
from docling.document_converter import DocumentConverter

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 【关键修改 1】定义全局信号量，限制并发数为 1 ---
# 对于 Mac/Ollama 本地部署，建议设置为 1 或 2。设置过高会导致 Ollama 崩溃。
embedding_semaphore = asyncio.Semaphore(1)

def init_lightrag(working_dir: str, llm_name: str = "qwen2.5:7b") -> LightRAG:
    """
    初始化 LightRAG，自动适配不同版本的参数 (embedding_function vs embedding_func)
    """
    lr_kwargs = {"working_dir": working_dir}
    
    # --- 【关键修改 2】重写 Embedding 实现，加入并发控制 ---
    async def _embedding_impl(texts: list[str]):
        # 使用信号量锁住请求，同一时间只允许 1 个请求进入 Ollama
        async with embedding_semaphore:
            try:
                # 注意：这里必须用 await，因为我们需要在锁的范围内等待结果返回
                return await ollama_embed(
                    texts, 
    embed_model="nomic-embed-text",
                    host="http://localhost:11434"
                )
            except Exception as e:
                logger.error(f"Embedding 请求失败: {e}")
                raise e

    # 2. 配置 LLM
    lr_kwargs["llm_model_func"] = ollama_model_complete # type: ignore
    lr_kwargs["llm_model_name"] = llm_name
    
    # 3. 尝试初始化 (兼容性处理)
    rag = None
    try:
        # 尝试方案 A: 新版参数 (embedding_function)
        logger.info("尝试使用新版参数 'embedding_function' 初始化...")
        rag = LightRAG(
            **lr_kwargs,
            embedding_function=_embedding_impl, # type: ignore
            embedding_dim=1024, # type: ignore
            # 这里的 batch_size 是指单次请求包含多少个文本块，不是并发数
            addon_params={"embedding_batch_size": 4} 
        )
    except TypeError as e:
        logger.warning(f"新版初始化失败 ({e})，尝试降级到旧版参数 'embedding_func'...")
        
        # 尝试方案 B: 旧版参数
        try:
            from lightrag.utils import EmbeddingFunc
        except ImportError:
            try:
                from lightrag import EmbeddingFunc # type: ignore
            except ImportError:
                logger.error("无法导入 EmbeddingFunc，初始化失败")
                raise e

        # 构造 EmbeddingFunc 对象
        embed_obj = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=_embedding_impl
        )
        
        rag = LightRAG(
            **lr_kwargs, # type: ignore
            embedding_func=embed_obj
        )

    # 4. 异步初始化存储
    async def _init_storages():
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
            await initialize_pipeline_status()
        except ImportError:
            pass

    try:
        asyncio.run(_init_storages())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_init_storages())
        loop.close()

    return rag

def process_file_with_docling(file_path: str) -> str:
    logger.info(f"[Docling] 开始处理文件: {file_path}")
    
    converter = DocumentConverter()
    result = converter.convert(file_path)
    md_content = result.document.export_to_markdown()
    
    logger.info(f"[Docling] 转换完成，生成了 {len(md_content)} 字符的 Markdown 内容")
    return md_content

def main():
    ap = argparse.ArgumentParser(description="Ingest PDF/Docx using Docling")
    ap.add_argument("--working_dir", required=True, help="LightRAG storage directory")
    ap.add_argument("--file", required=True, help="Path to the PDF or Docx file")
    ap.add_argument("--llm_name", default="qwen2.5:7b", help="Ollama model name")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"文件未找到: {args.file}")
        return

    # 1. 初始化 LightRAG
    logger.info(f"正在初始化 RAG 引擎 (Dir: {args.working_dir})...")
    try:
        rag = init_lightrag(args.working_dir, args.llm_name)
    except Exception as e:
        logger.error(f"RAG 引擎初始化失败: {e}")
        return

    # 2. 使用 Docling 处理文档
    try:
        content = process_file_with_docling(args.file)
    except Exception as e:
        logger.error(f"Docling 处理失败: {e}")
        return

    # 3. 入库
    if content.strip():
        logger.info("正在将 Markdown 内容写入 LightRAG (自动构建图谱和向量)...")
        try:
            # --- 【关键修改 3】增加 try-except 捕获入库时的异常 ---
            rag.insert(content)
            logger.info("✅ 入库成功！")
            
            print("\n--- 转换内容预览 ---")
            print(content[:500])
            print("...\n------------------\n")
        except Exception as e:
            logger.error(f"❌ 入库过程中发生严重错误: {e}")
            logger.error("建议：请检查 Ollama 服务是否存活 (ollama serve)，或尝试进一步降低并发。")
    else:
        logger.warning("转换后的内容为空，跳过入库。")

if __name__ == "__main__":
    main()