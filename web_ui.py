import streamlit as st
import requests
import json
import time

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="MineralRAG 问答",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 后端 API 地址
API_URL = "http://localhost:8000/v1/chat"

# --- 2. 侧边栏配置 ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/diamond.png", width=80)
    st.title("MineralRAG")
    st.markdown("---")
    
    st.subheader("⚙️ 检索增强设置")
    enable_web = st.toggle("🌐 联网搜索 (Web)", value=True, help="启用 DuckDuckGo 搜索实时信息")
    enable_graph = st.toggle("🕸️ 图谱推理 (Graph)", value=True, help="启用 Neo4j 知识图谱多跳推理")
    enable_vector = st.toggle("📄 文档检索 (Vector)", value=True, help="启用 Milvus 本地文档向量检索")
    
    st.markdown("---")
    st.info("💡 提示：这是一个基于 LangGraph + Milvus + Neo4j 的多源检索系统。")

# --- 3. 初始化会话状态 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. 渲染历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果历史消息里有 extra_info (来源和推理过程)，也可以选择渲染出来
        # 这里为了界面简洁，历史消息只显示文本，当次回答显示完整信息

# --- 5. 处理用户输入 ---
if prompt := st.chat_input("请输入关于矿物的问题 (例如: 石膏的用途是什么？)"):
    # 5.1 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5.2 请求后端并流式/块式显示
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 构造请求 Payload
        payload = {
            "query": prompt,
            "enable_vector": enable_vector,
            "enable_graph": enable_graph,
            "enable_web": enable_web
        }

        try:
            with st.spinner("🔍 Agent 正在进行多源检索与推理..."):
                start_time = time.time()
                response = requests.post(API_URL, json=payload)
                end_time = time.time()
                
            if response.status_code == 200:
                data = response.json()

                # --- A. 展示推理轨迹 (类似于 DeepSeek 的思考过程) ---
                trace = data.get("reasoning_trace", [])
                if trace:
                    with st.status("🧠 思考与规划过程", expanded=False) as status:
                        for step in trace:
                            st.write(step)
                        status.update(label=f"✅ 推理完成 (耗时 {data.get('latency', 0):.2f}s)", state="complete", expanded=False)

                # --- B. 展示最终回答 ---
                final_answer = data.get("answer", "未生成回答")
                message_placeholder.markdown(final_answer)
                
                # --- C. 展示来源卡片 ---
                sources = data.get("sources", [])
                if sources:
                    st.markdown("---")
                    st.subheader("📚 引用来源")
                    
                    # 使用 Tabs 分类展示，或者直接列出
                    # 这里我们根据 source_type 动态分配图标
                    for src in sources:
                        sType = src.get("source_type", "unknown")
                        
                        if sType == "web":
                            icon = "🌐"
                            title = "互联网来源"
                            color = "blue"
                        elif sType == "graph":
                            icon = "🕸️"
                            title = "知识图谱"
                            color = "purple"
                        elif sType == "vector":
                            icon = "📄"
                            title = "本地文档"
                            color = "green"
                        else:
                            icon = "❓"
                            title = "未知来源"
                            color = "grey"

                        # 截取内容预览
                        content_preview = src.get("content", "")
                        # 尝试从内容中提取标题 (例如 Web Source 通常有 Title: xxx)
                        display_title = title
                        if "Title:" in content_preview:
                            try:
                                display_title = content_preview.split("Title:")[1].split("\n")[0].strip()
                            except:
                                pass
                        
                        with st.expander(f"{icon} {display_title}"):
                            st.caption(f"来源类型: {sType}")
                            st.text(content_preview) # 使用 text 防止 markdown 渲染混乱
                            if src.get("metadata"):
                                st.json(src.get("metadata"))

                # 保存助手回复到历史
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            else:
                st.error(f"服务器错误: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"连接失败: {str(e)}")