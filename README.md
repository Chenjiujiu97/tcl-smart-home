TCL AI 智能家居虚拟管家 (TCL Smart Home AI Gateway)

本项目是“TCL 2026 AI 应用创新大赛”参赛作品。本项目旨在通过大模型技术（LLM Agent），打破传统智能家居“指令僵化、服务孤岛、缺乏售后深度”的困境，打造一个具备深度意图理解、本地 RAG 专家排障、长期记忆与商业价值转化能力的 AI 超级管家。

🌐 在线体验 (Demo)

评委与用户可直接通过以下链接在浏览器中体验演示效果，无需安装任何开发环境：
https://tcl-smart-home-auqybcx6nr7byrdgn65kwc.streamlit.app/

💡 核心功能亮点

专家级排障 (RAG)：无需人工客服，通过本地向量数据库精准回答各类家电报错（如 E1/E2 故障）。

环境自适应联动：基于用户当前情绪与意图，自主编排空调、灯光、窗帘等设备状态。

无感商业导购：将用户痛点转化为 TCL 爆款产品的精准营销话术。

隐私级物流引擎：通过长期记忆存储单号，配合静默鉴权实现隐私快递查询，无需用户输入繁琐信息。

🚀 本地快速复现指南

如果您希望在本地部署或二次开发本项目，请按以下步骤操作：

1. 环境准备

确保已安装 Python 3.9+ 环境，建议创建虚拟环境以防依赖冲突：

# 创建虚拟环境
python -m venv .venv

# 激活环境 (Windows)
.\.venv\Scripts\activate
# 激活环境 (Mac/Linux)
source .venv/bin/activate


2. 一键安装依赖

在项目根目录运行以下命令安装所有核心依赖：

pip install -r requirements.txt


3. 启动项目

执行以下命令即可启动服务，浏览器会自动弹出交互界面：

streamlit run app.py


🛠 技术栈

Agent 架构：LangGraph / LangChain

大模型：DeepSeek-Chat

RAG引擎：FAISS + HuggingFace (text2vec-base-chinese)

交互界面：Streamlit

语音合成：edge-tts
