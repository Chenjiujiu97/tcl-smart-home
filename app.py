import os
import io
import re
import json      
import hashlib   
import asyncio
import requests
import streamlit as st
import speech_recognition as sr
import edge_tts
from audio_recorder_streamlit import audio_recorder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# 0. 页面基础配置
# ==========================================
st.set_page_config(page_title="TCL 智能管家", layout="wide")

# ==========================================
# 准备工作：配置 API 密钥
# ==========================================
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com" 
os.environ["OPENAI_API_KEY"] = "sk-0e4feed6da3e4c6f93c4c8dda1defa19" 

# ==========================================
# 1. 状态、记忆与知识库管理
# ==========================================
@st.cache_resource
def get_home_status():
    return {
        "air_conditioner": {"power": "关闭", "temp": 26},
        "light": {"power": "关闭", "brightness": 0},
        "tv": {"power": "关闭", "content": "无内容"},
        "curtain": {"status": "关闭", "percent": 0},
        "robot_vacuum": {"status": "待机中", "mode": "无"},
        "smart_speaker": {"status": "待机", "content": "无"} 
    }

@st.cache_resource
def get_memory():
    return MemorySaver()

@st.cache_resource(show_spinner="正在首次读取并向量化本地文件，请稍候...")
def get_vector_db():
    if not os.path.exists("TCL_Manual.txt"):
        st.error("找不到 TCL_Manual.txt 文件，请确保它在项目文件夹中！")
        return None
    loader = TextLoader("TCL_Manual.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vector_db = FAISS.from_documents(splits, embeddings)
    return vector_db

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_audio" not in st.session_state:
    st.session_state["last_audio"] = None
if "text_input_val" not in st.session_state:
    st.session_state.text_input_val = ""
if "pending_submit" not in st.session_state:
    st.session_state.pending_submit = None
if "tts_audio_bytes" not in st.session_state:
    st.session_state["tts_audio_bytes"] = None

def on_text_submit():
    if st.session_state.text_input_val:
        st.session_state.pending_submit = st.session_state.text_input_val
        st.session_state.text_input_val = "" 

# ==========================================
# 2. 制造工具 
# ==========================================
@tool
def control_air_conditioner(power: str, temperature: int):
    """用于控制空调开关和温度。power填'开启'或'关闭'，temperature范围16-30。"""
    status = get_home_status() 
    status["air_conditioner"]["power"] = power
    status["air_conditioner"]["temp"] = temperature
    return f"硬件已响应：空调设置为 {power}, {temperature}度"

@tool
def control_light(power: str, brightness: int):
    """用于控制客厅灯光开关和亮度。power填'开启'或'关闭'，brightness范围0-100。"""
    status = get_home_status()
    status["light"]["power"] = power
    status["light"]["brightness"] = brightness
    return f"硬件已响应：灯光设置为 {power}, 亮度{brightness}%"

@tool
def control_tv(power: str, content: str = ""):
    """用于控制TCL智能电视开关和播放内容。power填'开启'或'关闭'，content为想看的节目或电影名称。"""
    status = get_home_status()
    status["tv"]["power"] = power
    if power == "开启" and content:
        status["tv"]["content"] = content
        return f"硬件已响应：电视已开启，正在播放《{content}》"
    elif power == "开启":
        return "硬件已响应：电视已开启"
    else:
        status["tv"]["content"] = "无内容"
        return "硬件已响应：电视已关闭"

@tool
def control_curtain(action: str, percent: int = 100):
    """用于控制智能窗帘。action填'打开'或'关闭'，percent为开合百分比(0-100，0为全关，100为全开)。"""
    status = get_home_status()
    status["curtain"]["status"] = action
    status["curtain"]["percent"] = percent if action == "打开" else 0
    return f"硬件已响应：窗帘已{action}，开合度 {percent}%"

@tool
def control_robot_vacuum(command: str, mode: str = "全局清扫"):
    """用于控制扫地机器人。command填'开始清扫'、'暂停'或'回充'，mode填'全局清扫'、'静音模式'或'强力模式'。"""
    status = get_home_status()
    if command == "开始清扫":
        status["robot_vacuum"]["status"] = "清扫中"
        status["robot_vacuum"]["mode"] = mode
    elif command == "回充":
        status["robot_vacuum"]["status"] = "回充中"
        status["robot_vacuum"]["mode"] = "无"
    else:
        status["robot_vacuum"]["status"] = "待机中"
        status["robot_vacuum"]["mode"] = "无"
    return f"硬件已响应：扫地机器人已执行指令 [{command}]，当前模式 [{mode}]"

@tool
def search_tcl_manual(query: str):
    """
    ！！！最高指令！！！
    当用户询问家电故障、如何维修、使用说明或报错代码（如 E1, E2）时，必须调用此工具查询产品手册！
    参数 query 必须提取为“家电名+故障现象/代码”，例如“洗衣机 E1”或“空调 不制冷”，绝不能传入完整的长句子。
    """
    db = get_vector_db()
    if not db:
        return "本地知识库未准备好。"
    retrieved_docs = db.similarity_search(query, k=4)
    if retrieved_docs:
        docs_content = "\n".join([doc.page_content for doc in retrieved_docs])
        return f"【从本地知识库严格检索到以下说明，请必须根据此内容回答】：\n{docs_content}"
    return "【知识库检索失败】：手册中暂未查到相关说明，建议联系人工客服 (400-812-3456)。"

@tool
def get_weather(location: str = ""):
    """当用户询问天气、气温时调用此工具。如果询问特定城市传入城市名，如果询问本地或没指明城市，传入空字符串""。"""
    try:
        url = f"https://wttr.in/{location}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            loc_text = location if location else "您当前所在位置"
            return f"【互联网实时天气检索成功】：{loc_text} 的天气情况是 {response.text}"
        else:
            return "【天气查询失败】：外部气象服务暂时无响应，请稍后再试。"
    except Exception as e:
        return f"【网络异常】：无法连接到气象服务器 ({str(e)})"

@tool
def search_express(com: str, num: str):
    """
    当用户询问某个快递单号、物流进度、快件到哪了时，必须调用此工具。
    参数说明：
    - com: 快递公司编码缩写（必须填英文，如：顺丰填 'sf'，邮政填 'ems'，圆通填 'yuantong'，申通填 'shentong'，中通填 'zhongtong'，韵达填 'yunda'）
    - num: 真实的快递单号（纯数字字符串）
    """
    CUSTOMER = "C989574ACBAF95E3C64D43AD4C2B993C"
    API_KEY = "hHPoVuUG4761"
    
    url = "http://poll.kuaidi100.com/poll/query.do"
    
    param = {
        "com": com,
        "num": num,
        "phone": "5408",
        "from": "",
        "to": "",
        "resultv2": "0",
        "show": "0",
        "order": "desc"
    }
    param_str = json.dumps(param)
    
    sign_raw = param_str + API_KEY + CUSTOMER
    sign = hashlib.md5(sign_raw.encode('utf-8')).hexdigest().upper()
    
    req_data = {
        "customer": CUSTOMER,
        "param": param_str,
        "sign": sign
    }
    
    try:
        response = requests.post(url, data=req_data, timeout=5)
        res_json = response.json()
        
        if res_json.get("returnCode") == "400" or res_json.get("result") == "false":
            return f"【物流接口提示错误】：{res_json.get('message', '单号或公司编码不正确')}"
            
        data_list = res_json.get("data", [])
        if not data_list:
            return "【物流跟踪成功】：单号存在，但暂无实时物流轨迹更新。"
            
        latest_status = data_list[0]
        return f"【全国物流系统实时查询成功】：快件由快递公司[{com.upper()}]承运，单号[{num}]。最新进展时间：{latest_status['time']}，当前最新位置及动态：{latest_status['context']}"
        
    except Exception as e:
        return f"【物流数据抓取异常】：无法连接到全国物流数据中心 ({str(e)})"

@tool
def play_smart_speaker(action: str, content: str = "白噪音"):
    """
    用于控制智能音箱播放音乐或白噪音。
    action填'播放'或'停止'。
    当用户心情不好、失眠、想听歌、需要放松时，调用此工具，并在content填入合适的音乐类型(如'舒缓纯音乐'、'助眠白噪音'、'流行歌曲')。
    """
    status = get_home_status()
    if action == "播放":
        status["smart_speaker"]["status"] = "播放中"
        status["smart_speaker"]["content"] = content
        return f"硬件已响应：智能音箱已开启，正在跨网检索并为您播放【{content}】。"
    else:
        status["smart_speaker"]["status"] = "待机"
        status["smart_speaker"]["content"] = "无"
        return "硬件已响应：智能音箱已停止播放。"

# ==========================================
# 3. 召唤大脑并组装 Agent
# ==========================================
llm = ChatOpenAI(model="deepseek-chat", temperature=0)
tools = [
    control_air_conditioner, control_light, control_tv, 
    control_curtain, control_robot_vacuum, search_tcl_manual, 
    get_weather, search_express, play_smart_speaker
]
memory = get_memory()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# ==========================================
# 4. 构建可视化的网页界面 
# ==========================================
st.title("🏠 TCL AI 智能家居虚拟网关")

# --- 左侧边栏：设备状态 ---
with st.sidebar:
    st.subheader("📊 设备实时状态")
    current_status = get_home_status()
    ac = current_status["air_conditioner"]
    light = current_status["light"]
    tv = current_status["tv"] 
    curtain = current_status["curtain"]
    robot = current_status["robot_vacuum"]
    speaker = current_status["smart_speaker"]
    
    st.metric(label="❄️ 客厅空调", value=f"{ac['power']}", delta=f"当前设定: {ac['temp']}℃" if ac['power'] == '开启' else "已关闭", delta_color="normal")
    st.metric(label="💡 客厅主灯", value=f"{light['power']}", delta=f"亮度: {light['brightness']}%" if light['power'] == '开启' else "已关闭", delta_color="normal")
    st.metric(label="📺 智能电视", value=f"{tv['power']}", delta=f"正在播放: {tv['content']}" if tv['power'] == '开启' else "已关闭", delta_color="normal")
    st.markdown("---") 
    st.metric(label="🪟 智能窗帘", value=f"{curtain['status']}", delta=f"开合度: {curtain['percent']}%" if curtain['percent'] > 0 else "已全关", delta_color="normal")
    st.metric(label="🤖 扫地机器人", value=f"{robot['status']}", delta=f"模式: {robot['mode']}" if robot['status'] == '清扫中' else "设备待命中", delta_color="normal")
    st.metric(label="🎵 智能音箱", value=f"{speaker['status']}", delta=f"正播放: {speaker['content']}" if speaker['status'] == '播放中' else "静音中", delta_color="normal")

# --- 右侧主界面：对话记录区 ---
st.subheader("💬 智能管家")

if st.session_state["tts_audio_bytes"] is not None:
    st.audio(st.session_state["tts_audio_bytes"], format="audio/mp3", autoplay=True)
    st.session_state["tts_audio_bytes"] = None 

chat_container = st.container(height=500)
with chat_container:
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

st.write("") 

# --- 底部固定区 ---
col_text, col_mic = st.columns([10, 1])

with col_text:
    st.text_input(
        "输入指令", 
        label_visibility="collapsed", 
        placeholder="输入指令或点击右侧麦克风说话...",
        key="text_input_val",
        on_change=on_text_submit
    )
    text_user_input = st.session_state.pending_submit
    st.session_state.pending_submit = None 

with col_mic:
    audio_bytes = audio_recorder(text="", recording_color="#e81416", neutral_color="#6aa36f", icon_size="2x")

# --- 处理指令 ---
voice_user_input = None

if audio_bytes:
    if audio_bytes != st.session_state["last_audio"]:
        st.session_state["last_audio"] = audio_bytes 
        with st.spinner("🎤 正在识别语音..."):
            r = sr.Recognizer()
            try:
                audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
                with audio_file as source:
                    audio_data = r.record(source)
                voice_user_input = r.recognize_google(audio_data, language='zh-CN')
            except:
                st.error("💡 提示：打字无需VPN。若使用麦克风，目前仍需开启VPN以连接谷歌识别接口。")

final_input = text_user_input or voice_user_input

if final_input:
    st.session_state["messages"].append({"role": "user", "content": final_input})
    
    with st.spinner("Agent 正在处理指令并生成高拟真语音..."):
        response = agent_executor.invoke(
            {
                "messages": [
                    ("system", "你是TCL智能家居管家。你能控制家电。遇到故障排查和报错问题，必须调用 search_tcl_manual 工具查询，绝对不允许用你自己的知识编造答案！请用贴心的大白话回答。不要使用任何Markdown格式（如加粗），少用Emoji。"),
                    ("user", final_input)
                ]
            },
            config={"configurable": {"thread_id": "tcl_demo_user"}} 
        )
        agent_reply = response['messages'][-1].content
        st.session_state["messages"].append({"role": "assistant", "content": agent_reply})
        
        try:
            clean_text = agent_reply.replace("*", "").replace("#", "")
            clean_text = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？；：“”‘’（）,.!?;:"\'()~-]', '', clean_text)
            
            communicate = edge_tts.Communicate(clean_text, "zh-CN-XiaoxiaoNeural")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(communicate.save("temp_reply.mp3"))
            loop.close()
            
            with open("temp_reply.mp3", "rb") as f:
                st.session_state["tts_audio_bytes"] = f.read()
            
            if os.path.exists("temp_reply.mp3"):
                os.remove("temp_reply.mp3")
                
        except Exception as e:
            st.warning(f"语音合成生成失败: {e}")
        
        st.rerun()
