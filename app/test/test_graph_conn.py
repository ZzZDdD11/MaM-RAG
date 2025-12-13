import sys
import os
import logging

# --- 1. 设置路径 ---
# 把项目根目录加入 Python 搜索路径，这样才能 import app
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 配置日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. 导入我们要测的模块 ---
from core.graph_store import get_graph_store

def test_neo4j_connection():
    print("\n" + "="*40)
    print("🚀 开始测试 Neo4j 连接...")
    print("="*40 + "\n")

    try:
        # 测试 1: 获取实例
        print("Step 1: 尝试获取 GraphStore 实例...")
        graph = get_graph_store()
        print("✅ 成功获取实例对象:", type(graph))

        # 测试 2: 验证单例模式
        print("\nStep 2: 验证单例模式 (再次获取)...")
        graph2 = get_graph_store()
        if graph is graph2:
            print("✅ 单例验证通过：两次获取的是同一个对象")
        else:
            print("❌ 单例验证失败：创建了新的对象！")

        # 测试 3: 执行实际查询 (Ping)
        print("\nStep 3: 执行 Cypher 查询 (Ping)...")
        # 刷新 Schema 是一个很好的连通性检查
        graph.refresh_schema() 
        schema = graph.schema
        print(f"✅ Schema 获取成功 (长度: {len(schema)} 字符)")
        
        # 执行一个简单的计算查询
        result = graph.query("RETURN 1 AS val")
        print(f"✅ 查询结果: {result}")
        
        if result and result[0]['val'] == 1:
            print("\n🎉🎉🎉 恭喜！Neo4j 连接完全正常！ 🎉🎉🎉")
        else:
            print("\n⚠️ 连接似乎成功，但查询结果不符合预期。")

    except Exception as e:
        print("\n❌ 测试失败！")
        print(f"错误详情: {e}")
        print("\n💡 排查建议:")
        print("1. 检查 Docker 容器是否运行: docker ps | grep neo4j")
        print("2. 检查端口是否开放: 7687 (Bolt协议)")
        print("3. 检查 app/core/config.py 中的账号密码是否匹配 docker-compose.yaml")

if __name__ == "__main__":
    test_neo4j_connection()