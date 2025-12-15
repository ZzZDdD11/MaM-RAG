from app.core.config import settings
import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
logger = logging.getLogger(__name__)

# 单例模式定义rerank模型
class RerankService:
    _instance = None
    _tokenizer = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._model is None:
            cls.init_model()


    @classmethod
    def init_model(cls):
        model_name = "BAAI/bge-reranker-base"
        logger.info(f"⚖️ 正在加载 Rerank 模型: {model_name} ...")

        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls._model.eval() # 开启评估模式
            
            # 如果有 GPU，使用 GPU
            if torch.cuda.is_available():
                cls._model.to('cuda')
            # Mac MPS (Metal Performance Shaders) 加速
            elif torch.backends.mps.is_available():
                cls._model.to('mps')
                
            logger.info("✅ Rerank 模型加载完成")
        except Exception as e:
            logger.error(f"❌ Rerank 模型加载失败: {e}")
            raise e
            
    @classmethod
    def compute_score(cls, query: str, documents: list[str]) -> list[float]: # type: ignore
        """
        计算查询与文档列表的相关性分数
        返回: 分数列表 (分数越高越相关)
        """
        if not documents:
            return []
        
        cls.get_instance()

        pairs = [[query,doc] for doc in documents]

        with torch.no_grad():
            inputs = cls._tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ) # type: ignore
            
            # 移动数据到设备
            if cls._model.device.type != 'cpu': # type: ignore
                inputs = {k: v.to(cls._model.device) for k, v in inputs.items()}
                
            scores = cls._model(**inputs, return_dict=True).logits.flatten().float() # type: ignore
            
            # 归一化分数 (可选，sigmoid 让分数在 0-1 之间)
            # scores = torch.sigmoid(scores) 
            
            return scores.cpu().tolist()
        
# 方便调用的函数
def rerank_documents(query: str, documents: list[str], top_k: int = 3):
    """
    输入查询和文档内容列表，返回重排序后的 Top-K 文档索引和分数
    """
    if not documents:
            return []
            
    scores = RerankService.compute_score(query, documents)
    # 将 (index, score) 结合并排序
    combined = list(enumerate(scores))
    # 按分数降序排列
    combined.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前 k 个结果: [(original_index, score), ...]
    return combined[:top_k]