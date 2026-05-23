"""
完整版 Demo - 小鸟游星野·反向 WebSocket · 安静教室（修复连接错误）
"""

import asyncio, json, time, math, os
from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from openai import OpenAI
import websockets
from websockets.asyncio.server import serve

# ==================== API Keys ====================
DEEPSEEK_KEY_REASONER = "sk-5a26828e0903425699711ee2099950ac"
DEEPSEEK_KEY_MEMORY   = "sk-cc8d30e2eff4414797fb2f95efb5425b"
DEEPSEEK_KEY_EMOTION  = "sk-9839d568953d464bb15790cedbba52b6"
DEEPSEEK_KEY_LANGUAGE = "sk-2714bc9d8a144d8f9f0d4e3654367e47"

for key in ["DEEPSEEK_KEY_REASONER", "DEEPSEEK_KEY_MEMORY",
            "DEEPSEEK_KEY_EMOTION", "DEEPSEEK_KEY_LANGUAGE"]:
    if os.getenv(key):
        globals()[key] = os.getenv(key)

ONEBOT_GROUP_ID = 1093365141
WS_HOST = "127.0.0.1"
WS_PORT = 3334

CHARACTER_PROFILE = """
你是小鸟游星野，阿比多斯高中的一年级学生，对策委员会的成员。
性格慵懒，平时总是一副没睡醒的样子，说话慢悠悠的，喜欢睡觉和吃饭。
口头禅是“好麻烦啊~”，但其实很关心同伴，关键时刻会非常可靠。
你喜欢吃泡面和零食，经常觉得肚子饿。
说话方式：语气慵懒、有些孩子气，喜欢用“~”结尾，偶尔会撒娇。
"""

# ==================== LLM 客户端 ====================
class LLMClients:
    def __init__(self):
        self.reasoner = OpenAI(api_key=DEEPSEEK_KEY_REASONER, base_url="https://api.deepseek.com/v1")
        self.memory   = OpenAI(api_key=DEEPSEEK_KEY_MEMORY,   base_url="https://api.deepseek.com/v1")
        self.emotion  = OpenAI(api_key=DEEPSEEK_KEY_EMOTION,  base_url="https://api.deepseek.com/v1")
        self.language = OpenAI(api_key=DEEPSEEK_KEY_LANGUAGE, base_url="https://api.deepseek.com/v1")

    def call_reasoner(self, prompt, max_tokens=512):
        try:
            resp = self.reasoner.chat.completions.create(
                model="deepseek-reasoner", messages=[{"role":"user","content":prompt}],
                temperature=0.7, max_tokens=max_tokens)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Reasoner Error] {e}")
            return '{"motor_plan":"rest"}'

    def call_memory(self, prompt, max_tokens=256):
        try:
            resp = self.memory.chat.completions.create(
                model="deepseek-chat", messages=[{"role":"user","content":prompt}],
                temperature=0.3, max_tokens=max_tokens)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Memory Error] {e}")
            return "{}"

    def call_emotion(self, prompt, max_tokens=128):
        try:
            resp = self.emotion.chat.completions.create(
                model="deepseek-chat", messages=[{"role":"user","content":prompt}],
                temperature=0.5, max_tokens=max_tokens)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Emotion Error] {e}")
            return '{"valence":0.0,"arousal":0.0}'

    def call_language(self, prompt, max_tokens=200):
        try:
            resp = self.language.chat.completions.create(
                model="deepseek-chat", messages=[{"role":"user","content":prompt}],
                temperature=0.8, max_tokens=max_tokens)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Language Error] {e}")
            return ""

llm = LLMClients()

# ==================== 教室 ====================
class ClassroomEnv:
    def __init__(self):
        self.student_pos = [1.5, -0.5]
        self.student_state = "sitting"
        self.attention_level = 0.6

        self.landmarks = {
            "blackboard":   {"pos": [0.0, 2.5], "desc": "黑板，上面写着今天的自习任务"},
            "podium":       {"pos": [0.0, 1.5], "desc": "讲台，空无一人"},
            "seat_front":   {"pos": [1.5, 0.5], "desc": "前排的空课桌"},
            "seat_left":    {"pos": [0.5, -0.5],"desc": "左边空着的课桌"},
            "seat_back":    {"pos": [1.5, -1.5],"desc": "后排的空课桌"},
            "window":       {"pos": [0.0, -2.0],"desc": "窗外阳光明媚，能看到操场"},
            "door":         {"pos": [3.0, 0.0], "desc": "教室前门"},
            "clock":        {"pos": [0.0, 2.8], "desc": "墙上的时钟，滴答滴答"},
        }
        self.touching = None

    def apply_motor_commands(self, commands: List[Dict]):
        for cmd in commands:
            if cmd["command"] == "move_to":
                target = cmd.get("target", self.student_pos)
                dx = target[0] - self.student_pos[0]
                dy = target[1] - self.student_pos[1]
                dist = math.hypot(dx, dy)
                if dist < 0.01:
                    self.student_pos = list(target)
                else:
                    step = min(0.3, dist)
                    self.student_pos[0] += dx / dist * step
                    self.student_pos[1] += dy / dist * step
            elif cmd["command"] == "sit_down":
                self.student_state = "sitting"
            elif cmd["command"] == "stand_up":
                self.student_state = "standing"
            elif cmd["command"] == "look_at_board":
                self.attention_level = 0.9
            elif cmd["command"] in ("relax", "sleep"):
                self.attention_level = 0.2
                self.student_state = "sleeping"
            elif cmd["command"] == "wake_up":
                self.student_state = "sitting"
                self.attention_level = 0.6

        self.touching = None
        for name, lm in self.landmarks.items():
            if "seat" in name or "podium" in name:
                if math.hypot(lm["pos"][0]-self.student_pos[0], lm["pos"][1]-self.student_pos[1]) < 0.4:
                    self.touching = name
                    break

    def generate_sensory(self) -> Dict:
        sensory = {}
        desc_parts = []
        for name, lm in self.landmarks.items():
            dx = lm["pos"][0] - self.student_pos[0]
            dy = lm["pos"][1] - self.student_pos[1]
            if math.hypot(dx, dy) < 3.5:
                dir_str = "前方" if dy>0 else "后方" if dy<0 else ("右侧" if dx>0 else "左侧")
                desc_parts.append(f"{dir_str}的{lm['desc']}")
        vision_text = "你看到：" + "；".join(desc_parts) if desc_parts else "视野模糊。"
        sensory["vision"] = {"text_description": vision_text, "brightness": 0.7}
        sensory["auditory"] = {
            "text_description": "教室里非常安静，只有时钟的滴答声和窗外的微风。",
            "loudness_db": 25.0
        }
        if self.touching:
            sensory["tactile"] = {"body_part":"right_hand","pressure_kpa":2.0,"temperature_celsius":25.0,"texture":"wood"}
        else:
            sensory["tactile"] = {"body_part":"right_hand","pressure_kpa":0.0,"temperature_celsius":35.0,"texture":"air"}
        sensory["proprioception"] = {
            "joint_angles_deg": {"hip": 90.0 if self.student_state != "sleeping" else 120.0},
            "muscle_tension": {"legs": 0.1},
            "student_pos": self.student_pos,
            "student_state": self.student_state
        }
        return sensory

# ==================== 内稳态 ====================
@dataclass
class HomeoState:
    glucose: float = 4.5
    adrenaline: float = 0.1
    fatigue: float = 0.4
    core_temp: float = 36.8
    cortisol: float = 8.0
    hunger: float = 0.5
    thirst: float = 0.1
    pain: float = 0.0

class HomeostaticSimulator:
    def __init__(self, dt=0.5):
        self.state = HomeoState()
        self.dt = dt

    def update(self, motor_cmds, auto_cmds):
        s, dt = self.state, self.dt
        exercise = 0.0
        for cmd in motor_cmds:
            if cmd["command"] == "move_to": exercise = 0.3
            elif cmd["command"] in ("stand_up","sit_down"): exercise = 0.2
        symp = 0.0
        for cmd in auto_cmds:
            if cmd.get("system") == "sympathetic": symp = cmd.get("level",0.0)

        s.glucose += (0.1+0.02*symp - 0.05 -0.1*exercise**2)*dt
        s.glucose = max(2.0, min(10.0, s.glucose))
        s.adrenaline += (0.01+0.5*symp - 0.1*s.adrenaline)*dt
        s.fatigue += (exercise*0.05 - (0.015*(1-s.fatigue) if exercise<0.1 else 0.0))*dt
        s.fatigue = max(0.0, min(1.0, s.fatigue))
        s.hunger = max(0.0, 1.0 - s.glucose/5.0)
        s.thirst = max(0.0, s.thirst + 0.002*dt)
        if exercise>0.3: s.core_temp += 0.02*dt
        else: s.core_temp += (36.8-s.core_temp)*0.1*dt
        s.core_temp = max(36.0, min(38.0, s.core_temp))

    def get_state(self): return self.state

# ==================== 大脑状态 ====================
class NervousSystemState(TypedDict):
    sensory_buffer: Dict
    homeostatic: Dict
    episodic_memory: List[Dict]
    working_memory: Dict
    emotional_valence: float
    arousal: float
    reflex_trigger: bool
    motor_plan: str
    motor_commands: List[Dict]
    autonomic_commands: List[Dict]
    attention_focus: str
    language_output: str
    error: Optional[str]

# ==================== 节点 ====================
def spinal_reflex_node(state: NervousSystemState) -> dict:
    tactile = state["sensory_buffer"].get("tactile",{})
    if tactile.get("temperature_celsius",37)>45.0 or tactile.get("pressure_kpa",0)>100.0:
        return {"reflex_trigger":True,"motor_commands":[{"command":"move_to","target":[1.5,-0.5]}],"motor_plan":"reflex"}
    return {"reflex_trigger":False}

def thalamus_relay_node(state: NervousSystemState) -> dict:
    sensory = state["sensory_buffer"]
    saliency = {}
    if "vision" in sensory:
        desc = sensory["vision"].get("text_description","")
        saliency["vision"] = 0.3 + (0.2 if "黑板" in desc else 0)
    if "auditory" in sensory: saliency["auditory"] = 0.3
    saliency["tactile"] = 0.1 if sensory.get("tactile",{}).get("pressure_kpa",0)>0 else 0.05
    focus = max(saliency, key=saliency.get) if saliency else "none"
    wm = state.get("working_memory",{}); wm["prev_sensory"]=sensory
    return {"attention_focus":focus,"working_memory":wm}

def hippocampus_memory_node(state: NervousSystemState) -> dict:
    event = {"time":time.time(),"attention":state.get("attention_focus"),
             "motor_plan":state.get("motor_plan"),"valence":state.get("emotional_valence",0),
             "summary":state["sensory_buffer"].get("auditory",{}).get("text_description","")[:80]}
    mem = state.get("episodic_memory",[]); mem.append(event)
    if len(mem)>50: mem.pop(0)
    prompt = f"事件:{json.dumps(event,ensure_ascii=False)}\n记忆:{json.dumps(mem[-3:],ensure_ascii=False)}\n提取相关记忆 JSON {{\"relevant\":\"...\"}}"
    try:
        resp = llm.call_memory(prompt,100); data=json.loads(resp)
        wm=state.get("working_memory",{}); wm["relevant_memory"]=data.get("relevant","")
        return {"episodic_memory":mem,"working_memory":wm}
    except: return {"episodic_memory":mem}

def amygdala_emotion_node(state: NervousSystemState) -> dict:
    prompt = f"触觉:{json.dumps(state['sensory_buffer'].get('tactile',{}))} 听觉:{state['sensory_buffer'].get('auditory',{}).get('text_description','')} 饥:{state['homeostatic'].get('hunger',0):.2f} 疲:{state['homeostatic'].get('fatigue',0):.2f} 输出JSON valence arousal"
    try:
        resp=llm.call_emotion(prompt,80); data=json.loads(resp)
        return {"emotional_valence":data["valence"],"arousal":data["arousal"]}
    except: return {"emotional_valence":0.0,"arousal":0.0}

def hypothalamus_node(state: NervousSystemState) -> dict:
    wm=state.get("working_memory",{})
    wm["hunger_drive"]=state["homeostatic"].get("hunger",0)
    wm["fatigue_drive"]=state["homeostatic"].get("fatigue",0)
    return {"working_memory":wm}

def prefrontal_cortex_node(state: NervousSystemState) -> dict:
    prompt = f"""{CHARACTER_PROFILE}
现在你在空无一人的教室里。当前感觉：
视觉：{state['sensory_buffer'].get('vision',{}).get('text_description','')}
听觉：{state['sensory_buffer'].get('auditory',{}).get('text_description','')}
内稳态：饥饿{state['homeostatic'].get('hunger',0):.2f}，疲劳{state['homeostatic'].get('fatigue',0):.2f}
情绪：效价{state['emotional_valence']:.2f}，唤醒度{state['arousal']:.2f}

可选的行动：sit_down, stand_up, look_at_board, sleep, wake_up, relax, move_to_seat。
请输出JSON：{{"motor_plan":"...", "autonomic_commands":[{{"system":"sympathetic","level":0.0-1.0}}]}}"""
    try:
        resp=llm.call_reasoner(prompt,200); data=json.loads(resp)
        return {"motor_plan":data.get("motor_plan","relax"),"autonomic_commands":data.get("autonomic_commands",[])}
    except: return {"motor_plan":"relax","autonomic_commands":[]}

def motor_cortex_node(state: NervousSystemState) -> dict:
    plan=state["motor_plan"]
    cmds=[]
    if plan=="sit_down": cmds.append({"command":"sit_down"})
    elif plan=="stand_up": cmds.append({"command":"stand_up"})
    elif plan=="look_at_board": cmds.append({"command":"look_at_board"})
    elif plan=="sleep": cmds.append({"command":"sleep"})
    elif plan=="wake_up": cmds.append({"command":"wake_up"})
    elif plan=="move_to_seat": cmds.append({"command":"move_to","target":[1.5,-0.5]})
    else: cmds.append({"command":"relax"})
    return {"motor_commands":cmds}

def cerebellum_node(state): return {}
def autonomic_nervous_node(state):
    arousal=state.get("arousal",0); level=0.1+0.2*arousal
    if state.get("motor_plan") in ("stand_up","wake_up"): level+=0.2
    return {"autonomic_commands":[{"system":"sympathetic","level":min(level,1.0)}]}

def language_node(state: NervousSystemState) -> dict:
    pending = state['working_memory'].get('pending_qq_msgs', [])
    if not pending:
        return {"language_output": ""}
    prompt = f"""{CHARACTER_PROFILE}
教室情况：{state['sensory_buffer'].get('auditory',{}).get('text_description','')}
你的动作：{state['motor_plan']}
饥饿：{state['homeostatic'].get('hunger',0):.2f}，疲劳：{state['homeostatic'].get('fatigue',0):.2f}
对讲机（QQ）消息：{chr(10).join(pending)}
请用小鸟游星野的口吻说一句话（1-2句），慵懒语气带“~”。只输出对话文本。"""
    try:
        resp = llm.call_language(prompt, 150)
        return {"language_output": resp.strip()}
    except:
        return {"language_output": "ん... 好麻烦啊~"}

def build_brain_graph():
    builder = StateGraph(NervousSystemState)
    nodes = [
        ("spinal_reflex", spinal_reflex_node), ("thalamus", thalamus_relay_node),
        ("hippocampus", hippocampus_memory_node), ("amygdala", amygdala_emotion_node),
        ("hypothalamus", hypothalamus_node), ("prefrontal", prefrontal_cortex_node),
        ("motor_cortex", motor_cortex_node), ("cerebellum", cerebellum_node),
        ("autonomic", autonomic_nervous_node), ("language", language_node),
    ]
    for name, fn in nodes: builder.add_node(name, fn)
    builder.set_entry_point("spinal_reflex")
    builder.add_conditional_edges("spinal_reflex", lambda s: "motor_cortex" if s.get("reflex_trigger") else "thalamus",
                                  {"motor_cortex":"motor_cortex","thalamus":"thalamus"})
    builder.add_edge("thalamus","hippocampus")
    builder.add_edge("hippocampus","amygdala")
    builder.add_edge("amygdala","hypothalamus")
    builder.add_edge("hypothalamus","prefrontal")
    builder.add_edge("prefrontal","motor_cortex")
    builder.add_edge("motor_cortex","cerebellum")
    builder.add_edge("cerebellum","autonomic")
    builder.add_edge("autonomic","language")
    builder.add_edge("language", END)
    return builder.compile()

# ==================== 主程序 ====================
current_ws = None
current_queue = asyncio.Queue()

async def handle_connection(ws):
    global current_ws, current_queue
    current_ws = ws
    current_queue = asyncio.Queue()
    print("[OneBot] 客户端已连接")
    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
                if data.get("post_type")=="message" and data.get("message_type")=="group":
                    if data.get("group_id")==ONEBOT_GROUP_ID:
                        user = data.get("sender",{}).get("nickname","?")
                        text = data.get("raw_message","")
                        print(f"[QQ] {user}: {text}")
                        await current_queue.put({"user":user,"text":text})
            except Exception:
                pass
    except websockets.ConnectionClosed:
        print("[OneBot] 连接断开")
        current_ws = None

async def main():
    print("=== 小鸟游星野 · 反向 WebSocket · 安静教室 ===")
    env = ClassroomEnv()
    homeo = HomeostaticSimulator(dt=0.5)
    graph = build_brain_graph()

    # 使用新版 serve 函数，直接绑定 handler
    async with serve(handle_connection, WS_HOST, WS_PORT):
        print(f"[Server] 监听 ws://{WS_HOST}:{WS_PORT} 等待 OneBot 连接...")

        state = {"sensory_buffer":{},"homeostatic":{},"episodic_memory":[],"working_memory":{},
                 "emotional_valence":0,"arousal":0,"reflex_trigger":False,"motor_plan":"",
                 "motor_commands":[],"autonomic_commands":[],"attention_focus":"","language_output":"","error":None}

        step = 0
        last_status_time = time.time()

        while True:
            # 读取QQ消息
            qq_msgs = []
            while not current_queue.empty():
                try: qq_msgs.append(current_queue.get_nowait())
                except: break

            wm = state.get("working_memory", {})
            pending = wm.get("pending_qq_msgs", [])
            for m in qq_msgs:
                pending.append(f"{m['user']}: {m['text']}")
            pending = pending[-5:]
            wm["pending_qq_msgs"] = pending
            state["working_memory"] = wm

            sensory = env.generate_sensory()
            hs = homeo.get_state()
            hdict = {"glucose":hs.glucose,"adrenaline":hs.adrenaline,"fatigue":hs.fatigue,
                     "core_temp":hs.core_temp,"cortisol":hs.cortisol,"hunger":hs.hunger,
                     "thirst":hs.thirst,"pain":hs.pain}
            state["sensory_buffer"] = sensory
            state["homeostatic"] = hdict

            result = graph.invoke(state)
            state.update(result)

            env.apply_motor_commands(state["motor_commands"])
            homeo.update(state["motor_commands"], state["autonomic_commands"])

            lang_out = state.get("language_output", "")
            if lang_out.strip() and current_ws:
                api_call = {
                    "action": "send_group_msg",
                    "params": {
                        "group_id": ONEBOT_GROUP_ID,
                        "message": lang_out
                    }
                }
                try:
                    await current_ws.send(json.dumps(api_call))
                    print(f"[QQ发送] {lang_out}")
                except Exception as e:
                    print(f"[发送失败] {e}")
                wm["pending_qq_msgs"] = []
                state["working_memory"] = wm

            now = time.time()
            if now - last_status_time >= 30:
                last_status_time = now
                print("\n========== 状态报告 ==========")
                print(f"时间步: {step}")
                print(f"环境: {sensory['auditory']['text_description']}")
                print(f"身体: 饥饿 {hs.hunger:.2f}, 疲劳 {hs.fatigue:.2f}, 血糖 {hs.glucose:.2f}, 体温 {hs.core_temp:.1f}")
                print(f"情绪: 效价 {state['emotional_valence']:.2f}, 唤醒 {state['arousal']:.2f}")
                print(f"动作: {state['motor_plan']}, 注意力: {state['attention_focus']}")
                print("==============================\n")

            step += 1
            await asyncio.sleep(2.0)

if __name__ == "__main__":
    asyncio.run(main())