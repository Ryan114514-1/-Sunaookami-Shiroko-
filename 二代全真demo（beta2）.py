#!/usr/bin/env python3
"""
Sunaookami Shiroko - 生物神经系统模拟系统
完整细化版 Demo (修复 JSON 解析、WebSocket 绑定、类型转换)
基于项目文档 v1.1 实现
"""

import asyncio
import json
import time
import math
import os
import random
import pickle
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, TypedDict

import numpy as np
from langgraph.graph import StateGraph, END
from openai import OpenAI
import websockets
from websockets.asyncio.server import serve

# ==================== API 密钥配置 ====================
DEEPSEEK_KEY_REASONER = "sk-4377a**********985f0b4bbb204595b"
DEEPSEEK_KEY_MEMORY   = "sk-d858***********8b4382a1ec65d9766"
DEEPSEEK_KEY_EMOTION  = "sk-7464*********************200f3ac"
DEEPSEEK_KEY_LANGUAGE = "sk-808f6d9bc1584*********c5158e20d6"

for key in ["DEEPSEEK_KEY_REASONER", "DEEPSEEK_KEY_MEMORY",
            "DEEPSEEK_KEY_EMOTION", "DEEPSEEK_KEY_LANGUAGE"]:
    if os.getenv(key):
        globals()[key] = os.getenv(key)

ONEBOT_GROUP_ID = 1093365141
WS_HOST = "127.0.0.1"          # 改为 0.0.0.0 允许外部连接
WS_PORT = 3334

CHARACTER_PROFILE = """
你是小鸟游星野，阿比多斯高中的一年级学生，对策委员会的成员。
性格慵懒，平时总是一副没睡醒的样子，说话慢悠悠的，喜欢睡觉和吃饭。
口头禅是“好麻烦啊~”，但其实很关心同伴，关键时刻会非常可靠。
你喜欢吃泡面和零食，经常觉得肚子饿。
说话方式：语气慵懒、有些孩子气，喜欢用“~”结尾，偶尔会撒娇。
"""

# ==================== LLM 多客户端 ====================
class LLMClients:
    def __init__(self):
        self.reasoner = OpenAI(api_key=DEEPSEEK_KEY_REASONER, base_url="https://api.deepseek.com/v1")
        self.memory   = OpenAI(api_key=DEEPSEEK_KEY_MEMORY,   base_url="https://api.deepseek.com/v1")
        self.emotion  = OpenAI(api_key=DEEPSEEK_KEY_EMOTION,  base_url="https://api.deepseek.com/v1")
        self.language = OpenAI(api_key=DEEPSEEK_KEY_LANGUAGE, base_url="https://api.deepseek.com/v1")

    def call_reasoner(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            resp = self.reasoner.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Reasoner Error] {e}")
            return '{"motor_plan":"rest"}'

    def call_memory(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            resp = self.memory.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Memory Error] {e}")
            return "{}"

    def call_emotion(self, prompt: str, max_tokens: int = 128) -> str:
        try:
            resp = self.emotion.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Emotion Error] {e}")
            return '{"valence":0.0,"arousal":0.0}'

    def call_language(self, prompt: str, max_tokens: int = 200) -> str:
        try:
            resp = self.language.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Language Error] {e}")
            return ""

llm = LLMClients()

# ==================== 感觉输入模拟系统（dataclass 定义）====================
@dataclass
class VisionData:
    timestamp_us: int
    text_description: str = ""
    features: Optional[List[float]] = None
    image_shape: Tuple[int, int] = (224, 224)
    field_of_view_deg: float = 120.0
    head_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    brightness: float = 0.7

@dataclass
class AuditoryData:
    timestamp_us: int
    text_description: str = ""
    mfcc: Optional[List[float]] = None
    direction_azimuth_deg: float = 0.0
    loudness_db: float = 25.0

@dataclass
class TactileData:
    timestamp_us: int
    body_part: str = "right_hand"
    pressure_kpa: float = 0.0
    temperature_celsius: float = 35.0
    texture: str = "smooth"
    vibration_freq_hz: float = 0.0

@dataclass
class OlfactoryData:
    timestamp_us: int
    odorant: str = "none"
    concentration_ppm: float = 0.0

@dataclass
class GustatoryData:
    timestamp_us: int
    taste_quality: str = "none"
    intensity: float = 0.0

@dataclass
class ProprioceptionData:
    timestamp_us: int
    joint_angles_deg: Dict[str, float] = field(default_factory=dict)
    muscle_tension: Dict[str, float] = field(default_factory=dict)
    linear_acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    student_pos: Tuple[float, float] = (0.0, 0.0)
    student_state: str = "sitting"

class SensorySimulator:
    def __init__(self, env, config: Dict = None):
        self.env = env
        self.config = config or {
            "vision_freq_hz": 30,
            "tactile_freq_hz": 100,
            "proprio_freq_hz": 100,
            "auditory_freq_hz": 20,
            "olfactory_freq_hz": 1,
            "noise_std": 0.05,
            "latency_sec": 0.02,
        }
        self.last_update = time.time()

    def _add_noise(self, value: float, noise_std: float = 0.05) -> float:
        return value + random.gauss(0, noise_std)

    async def get_vision(self) -> VisionData:
        desc_parts = []
        for name, lm in self.env.landmarks.items():
            dx = lm["pos"][0] - self.env.student_pos[0]
            dy = lm["pos"][1] - self.env.student_pos[1]
            if math.hypot(dx, dy) < 3.5:
                dir_str = "前方" if dy>0 else "后方" if dy<0 else ("右侧" if dx>0 else "左侧")
                desc_parts.append(f"{dir_str}的{lm['desc']}")
        vision_text = "你看到：" + ("；".join(desc_parts) if desc_parts else "视野模糊。")
        features = [self._add_noise(random.random(), 0.1) for _ in range(512)]
        return VisionData(
            timestamp_us=int(time.time() * 1e6),
            text_description=vision_text,
            features=features,
            brightness=self._add_noise(self.env.lighting, 0.05),
            head_orientation=(0.0, self.env.head_yaw, 0.0)
        )

    async def get_auditory(self) -> AuditoryData:
        text = "教室里非常安静，只有时钟的滴答声和窗外的微风。"
        if self.env.student_state == "sleeping":
            text = "你隐约听到自己的呼吸声，周围很安静。"
        direction = 0.0
        loudness = 25.0 + random.gauss(0, 2)
        mfcc = [self._add_noise(0.5, 0.2) for _ in range(12)]
        return AuditoryData(
            timestamp_us=int(time.time() * 1e6),
            text_description=text,
            mfcc=mfcc,
            direction_azimuth_deg=direction,
            loudness_db=min(100, max(0, self._add_noise(loudness, 1.0)))
        )

    async def get_tactile(self) -> TactileData:
        if self.env.touching is None:
            return TactileData(
                timestamp_us=int(time.time() * 1e6),
                body_part="right_hand",
                pressure_kpa=0.0,
                temperature_celsius=35.0,
                texture="air"
            )
        obj = self.env.touching
        if "seat" in obj:
            pressure = 2.0
            texture = "wood" if "desk" not in obj else "cloth"
            temp = 25.0
        elif "podium" in obj:
            pressure = 5.0
            texture = "wood"
            temp = 24.0
        else:
            pressure = 1.0
            texture = "rough"
            temp = 28.0
        return TactileData(
            timestamp_us=int(time.time() * 1e6),
            body_part="right_hand",
            pressure_kpa=self._add_noise(pressure, 0.1),
            temperature_celsius=self._add_noise(temp, 0.5),
            texture=texture
        )

    async def get_olfactory(self) -> OlfactoryData:
        return OlfactoryData(
            timestamp_us=int(time.time() * 1e6),
            odorant="none",
            concentration_ppm=0.0
        )

    async def get_gustatory(self) -> GustatoryData:
        return GustatoryData(
            timestamp_us=int(time.time() * 1e6),
            taste_quality="none",
            intensity=0.0
        )

    async def get_proprioception(self) -> ProprioceptionData:
        joint_angles = {"hip": 90.0, "knee": 170.0, "shoulder": 10.0}
        if self.env.student_state == "sitting":
            joint_angles["hip"] = 90.0
            joint_angles["knee"] = 90.0
        elif self.env.student_state == "standing":
            joint_angles["hip"] = 0.0
            joint_angles["knee"] = 0.0
        elif self.env.student_state == "sleeping":
            joint_angles["hip"] = 120.0
            joint_angles["knee"] = 20.0
        muscle_tension = {"legs": 0.1, "back": 0.2, "arms": 0.05}
        if self.env.student_state == "standing":
            muscle_tension["legs"] = 0.4
        return ProprioceptionData(
            timestamp_us=int(time.time() * 1e6),
            joint_angles_deg=joint_angles,
            muscle_tension=muscle_tension,
            linear_acceleration=(0.0, -9.8, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            student_pos=self.env.student_pos,
            student_state=self.env.student_state
        )

    async def gather_all(self) -> Dict[str, Any]:
        vision_task = asyncio.create_task(self.get_vision())
        auditory_task = asyncio.create_task(self.get_auditory())
        tactile_task = asyncio.create_task(self.get_tactile())
        proprio_task = asyncio.create_task(self.get_proprioception())
        olfactory_task = asyncio.create_task(self.get_olfactory())
        gustatory_task = asyncio.create_task(self.get_gustatory())
        await asyncio.sleep(self.config["latency_sec"])
        vision, auditory, tactile, proprio, olfactory, gustatory = await asyncio.gather(
            vision_task, auditory_task, tactile_task, proprio_task, olfactory_task, gustatory_task
        )
        return {
            "vision": vision,
            "auditory": auditory,
            "tactile": tactile,
            "proprioception": proprio,
            "olfactory": olfactory,
            "gustatory": gustatory,
        }

# ==================== 身体内稳态模拟系统 ====================
@dataclass
class HomeostaticState:
    glucose: float = 5.0
    liver_glycogen: float = 100.0
    free_fatty_acids: float = 0.5
    blood_osmolarity: float = 290.0
    adh: float = 2.0
    core_temp: float = 37.0
    skin_temp: float = 33.0
    cortisol: float = 10.0
    adrenaline: float = 0.1
    noradrenaline: float = 0.2
    muscle_fatigue: float = 0.1
    central_fatigue: float = 0.1
    hunger: float = 0.3
    thirst: float = 0.1
    sleep_pressure: float = 0.2
    pain_level: float = 0.0

class HomeostaticSimulator:
    def __init__(self, dt: float = 0.1):
        self.state = HomeostaticState()
        self.dt = dt
        self.time = 0.0

    def update(self, motor_commands: List[Dict], autonomic_commands: List[Dict]):
        s = self.state
        dt = self.dt

        exercise_intensity = 0.0
        for cmd in motor_commands:
            if cmd.get("command") == "move_to":
                exercise_intensity = max(exercise_intensity, 0.3)
            elif cmd.get("command") in ("stand_up", "sit_down"):
                exercise_intensity = max(exercise_intensity, 0.2)
            elif cmd.get("command") in ("run", "jump"):
                exercise_intensity = max(exercise_intensity, 0.7)
        sympathetic_activation = 0.0
        for cmd in autonomic_commands:
            if cmd.get("system") == "sympathetic":
                sympathetic_activation = max(sympathetic_activation, cmd.get("level", 0.0))

        glucose_prod = 0.1 + 0.02 * sympathetic_activation
        glucose_consume = 0.05 + 0.1 * (exercise_intensity ** 2)
        s.glucose += (glucose_prod - glucose_consume) * dt
        s.glucose = np.clip(s.glucose, 2.0, 10.0)

        adrenaline_prod = 0.01 + 0.5 * sympathetic_activation
        s.adrenaline += (adrenaline_prod - 0.1 * s.adrenaline) * dt
        s.adrenaline = np.clip(s.adrenaline, 0.0, 2.0)

        cortisol_prod = 0.02 + 0.3 * sympathetic_activation
        s.cortisol += (cortisol_prod - 0.05 * s.cortisol) * dt
        s.cortisol = np.clip(s.cortisol, 5.0, 40.0)

        fatigue_accum = exercise_intensity * 0.05
        if exercise_intensity < 0.1:
            fatigue_recovery = 0.08 * (1.0 - s.muscle_fatigue)
        else:
            fatigue_recovery = 0.0
        s.muscle_fatigue += (fatigue_accum - fatigue_recovery) * dt
        s.muscle_fatigue = np.clip(s.muscle_fatigue, 0.0, 1.0)

        central_fatigue_accum = exercise_intensity * 0.02 + s.sleep_pressure * 0.005
        s.central_fatigue += (central_fatigue_accum - 0.01 * s.central_fatigue) * dt
        s.central_fatigue = np.clip(s.central_fatigue, 0.0, 1.0)

        heat_prod = 0.05 + 0.2 * exercise_intensity
        if s.core_temp < 36.0:
            heat_prod += 0.1 * (36.0 - s.core_temp)
        heat_loss = 0.1 * (s.core_temp - 25.0) + 0.05 * max(0, s.core_temp - 36.5) ** 2
        s.core_temp += (heat_prod - heat_loss) * dt
        s.core_temp = np.clip(s.core_temp, 35.0, 39.0)
        s.skin_temp += 0.2 * (s.core_temp - s.skin_temp) * dt

        s.hunger = max(0.0, 1.0 - (s.glucose + s.liver_glycogen/200) / 6.0)
        s.hunger = np.clip(s.hunger, 0.0, 1.0)

        s.blood_osmolarity = 285.0 + 5.0 * s.thirst
        s.thirst = min(1.0, s.thirst + 0.002 * dt)
        if s.thirst > 0.8:
            s.adh = 8.0

        if motor_commands and any(cmd.get("command") == "sleep" for cmd in motor_commands):
            s.sleep_pressure = max(0.0, s.sleep_pressure - 0.01 * dt)
        else:
            s.sleep_pressure = min(1.0, s.sleep_pressure + 0.002 * dt)

        s.pain_level = max(0.0, s.pain_level - 0.05 * dt)

        s.glucose += random.gauss(0, 0.01) * dt
        s.adrenaline += random.gauss(0, 0.005) * dt
        s.core_temp += random.gauss(0, 0.02) * dt

    def get_state_dict(self) -> Dict[str, float]:
        return asdict(self.state)

    def apply_pain(self, intensity: float):
        self.state.pain_level = min(1.0, self.state.pain_level + intensity)

# ==================== 神经系统状态定义 ====================
class NervousSystemState(TypedDict):
    sensory_buffer: Dict[str, Any]
    attention_focus: Optional[str]
    homeostatic_state: Dict[str, float]
    episodic_memory: List[Dict]
    working_memory: Dict[str, Any]
    emotional_valence: float
    arousal: float
    reflex_trigger: bool
    motor_plan: str
    motor_commands: List[Dict[str, Any]]
    autonomic_commands: List[Dict[str, Any]]
    language_output: str
    error: Optional[str]
    timestamp: float

# ==================== 环境（教室）====================
class ClassroomEnv:
    def __init__(self):
        self.student_pos = [1.5, -0.5]
        self.student_state = "sitting"
        self.head_yaw = 0.0
        self.lighting = 0.7
        self.touching = None
        self.landmarks = {
            "blackboard":   {"pos": [0.0, 2.5], "desc": "黑板，上面写着今天的自习任务"},
            "podium":       {"pos": [0.0, 1.5], "desc": "讲台，空无一人"},
            "seat_front":   {"pos": [1.5, 0.5], "desc": "前排的空课桌"},
            "seat_left":    {"pos": [0.5, -0.5], "desc": "左边空着的课桌"},
            "seat_back":    {"pos": [1.5, -1.5], "desc": "后排的空课桌"},
            "window":       {"pos": [0.0, -2.0], "desc": "窗外阳光明媚，能看到操场"},
            "door":         {"pos": [3.0, 0.0], "desc": "教室前门"},
            "clock":        {"pos": [0.0, 2.8], "desc": "墙上的时钟，滴答滴答"},
        }

    def apply_motor_commands(self, commands: List[Dict]):
        for cmd in commands:
            if cmd["command"] == "move_to":
                target = cmd.get("target", self.student_pos)
                dx = target[0] - self.student_pos[0]
                dy = target[1] - self.student_pos[1]
                dist = math.hypot(dx, dy)
                if dist < 0.05:
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
                self.head_yaw = 0.0
                self.lighting = 0.9
            elif cmd["command"] == "sleep":
                self.student_state = "sleeping"
                self.lighting = 0.1
            elif cmd["command"] == "wake_up":
                self.student_state = "sitting"
                self.lighting = 0.7
            elif cmd["command"] == "relax":
                self.student_state = "sitting"
                self.head_yaw = random.uniform(-30, 30)

        self.touching = None
        for name, lm in self.landmarks.items():
            if "seat" in name or "podium" in name:
                if math.hypot(lm["pos"][0]-self.student_pos[0], lm["pos"][1]-self.student_pos[1]) < 0.4:
                    self.touching = name
                    break

# ==================== 辅助函数：鲁棒 JSON 解析 ====================
def extract_json_from_llm_response(response: str) -> dict:
    """从 LLM 响应中提取 JSON 对象，处理截断或不完整情况"""
    if not response:
        return {}
    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # 使用正则匹配第一个 { ... } 对象
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        json_str = match.group(0)
        # 尝试补全不完整的 JSON（简单处理：如果最后一个字符不是 }，尝试添加）
        if not json_str.endswith('}'):
            # 找到最后一个完整的键值对
            pass
        try:
            return json.loads(json_str)
        except:
            pass
    # 尝试从截断的 JSON 中提取可用的字段
    result = {}
    # 匹配 "key": value 或 "key": "value"
    for pattern in [r'"(\w+)"\s*:\s*"([^"]*)"', r'"(\w+)"\s*:\s*([\d\.\-]+)']:
        for match in re.finditer(pattern, response):
            key, val = match.groups()
            if val.replace('.', '').replace('-', '').isdigit():
                result[key] = float(val) if '.' in val else int(val)
            else:
                result[key] = val
    return result

# ==================== LangGraph 节点（修正属性访问 + 鲁棒解析）====================
def spinal_reflex_node(state: NervousSystemState) -> dict:
    tactile = state["sensory_buffer"].get("tactile")
    if tactile:
        if tactile.temperature_celsius > 45.0 or tactile.pressure_kpa > 100.0:
            return {
                "reflex_trigger": True,
                "motor_commands": [{"command": "move_to", "target": [1.5, -0.5]}],
                "motor_plan": "reflex_withdraw"
            }
    pain = state["homeostatic_state"].get("pain_level", 0.0)
    if pain > 0.7:
        return {
            "reflex_trigger": True,
            "motor_commands": [{"command": "move_to", "target": [1.5, -0.5]}],
            "motor_plan": "pain_avoid"
        }
    return {"reflex_trigger": False}

def thalamus_relay_node(state: NervousSystemState) -> dict:
    sensory = state["sensory_buffer"]
    saliency = {}
    vision = sensory.get("vision")
    if vision:
        if "黑板" in vision.text_description:
            saliency["vision"] = 0.7
        elif "时钟" in vision.text_description:
            saliency["vision"] = 0.4
        else:
            saliency["vision"] = 0.3
    auditory = sensory.get("auditory")
    if auditory:
        if "时钟" in auditory.text_description:
            saliency["auditory"] = 0.2
        else:
            saliency["auditory"] = 0.1
    tactile = sensory.get("tactile")
    if tactile:
        if tactile.pressure_kpa > 0:
            saliency["tactile"] = 0.6
        else:
            saliency["tactile"] = 0.05
    proprio = sensory.get("proprioception")
    if proprio:
        prev_state = state.get("working_memory", {}).get("prev_state")
        if proprio.student_state != prev_state:
            saliency["proprioception"] = 0.5
    focus = max(saliency, key=saliency.get) if saliency else "none"
    wm = state.get("working_memory", {})
    if proprio:
        wm["prev_state"] = proprio.student_state
    wm["prev_sensory"] = {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v for k, v in sensory.items()}
    return {"attention_focus": focus, "working_memory": wm}

def hippocampus_memory_node(state: NervousSystemState) -> dict:
    auditory = state["sensory_buffer"].get("auditory")
    summary = auditory.text_description[:80] if auditory else ""
    event = {
        "time": time.time(),
        "attention": state.get("attention_focus"),
        "motor_plan": state.get("motor_plan"),
        "valence": state.get("emotional_valence", 0),
        "summary": summary
    }
    mem = state.get("episodic_memory", [])
    mem.append(event)
    if len(mem) > 50:
        mem.pop(0)
    prompt = f"当前事件: {json.dumps(event, ensure_ascii=False)}\n最近记忆: {json.dumps(mem[-5:], ensure_ascii=False)}\n请提取一条与当前情景最相关的记忆（一句话）。输出 JSON: {{\"relevant\": \"...\"}}"
    try:
        resp = llm.call_memory(prompt, max_tokens=100)
        data = extract_json_from_llm_response(resp)
        wm = state.get("working_memory", {})
        wm["relevant_memory"] = data.get("relevant", "")
        return {"episodic_memory": mem, "working_memory": wm}
    except:
        return {"episodic_memory": mem}

def amygdala_emotion_node(state: NervousSystemState) -> dict:
    tactile = state["sensory_buffer"].get("tactile")
    tactile_str = f"pressure={tactile.pressure_kpa:.1f}, temp={tactile.temperature_celsius:.1f}" if tactile else "none"
    auditory = state["sensory_buffer"].get("auditory")
    audio_text = auditory.text_description if auditory else ""
    hs = state["homeostatic_state"]
    prompt = f"""
触觉: {tactile_str}
听觉: {audio_text}
饥饿: {hs.get('hunger', 0):.2f}
疲劳: {hs.get('muscle_fatigue', 0):.2f}
疼痛: {hs.get('pain_level', 0):.2f}
肾上腺素: {hs.get('adrenaline', 0):.2f}
请输出情绪效价 valence（-1 消极到1积极）和唤醒度 arousal（0 平静到1激动）。
输出 JSON: {{"valence": float, "arousal": float}}
"""
    try:
        resp = llm.call_emotion(prompt, max_tokens=80)
        data = extract_json_from_llm_response(resp)
        valence = np.clip(float(data.get("valence", 0.0)), -1.0, 1.0)
        arousal = np.clip(float(data.get("arousal", 0.0)), 0.0, 1.0)
        return {"emotional_valence": valence, "arousal": arousal}
    except:
        return {"emotional_valence": 0.0, "arousal": 0.0}

def hypothalamus_node(state: NervousSystemState) -> dict:
    wm = state.get("working_memory", {})
    hs = state["homeostatic_state"]
    wm["hunger_drive"] = hs.get("hunger", 0)
    wm["fatigue_drive"] = hs.get("muscle_fatigue", 0)
    wm["sleep_pressure"] = hs.get("sleep_pressure", 0)
    wm["thirst_drive"] = hs.get("thirst", 0)
    if hs.get("hunger", 0) > 0.7:
        wm["urgent_need"] = "eat"
    elif hs.get("thirst", 0) > 0.6:
        wm["urgent_need"] = "drink"
    else:
        wm["urgent_need"] = "none"
    return {"working_memory": wm}

def prefrontal_cortex_node(state: NervousSystemState) -> dict:
    wm = state.get("working_memory", {})
    vision = state["sensory_buffer"].get("vision")
    vision_text = vision.text_description if vision else ""
    auditory = state["sensory_buffer"].get("auditory")
    audio_text = auditory.text_description if auditory else ""
    tactile = state["sensory_buffer"].get("tactile")
    tactile_str = f"压力 {tactile.pressure_kpa:.1f} kPa" if tactile else "无"
    hs = state["homeostatic_state"]
    prompt = f"""{CHARACTER_PROFILE}
现在你在空无一人的教室里。当前感觉：
视觉：{vision_text}
听觉：{audio_text}
触觉：{tactile_str}
内稳态：
- 饥饿: {hs.get('hunger', 0):.2f}
- 疲劳: {hs.get('muscle_fatigue', 0):.2f}
- 睡眠压力: {hs.get('sleep_pressure', 0):.2f}
- 肾上腺素: {hs.get('adrenaline', 0):.2f}
情绪：效价 {state['emotional_valence']:.2f}, 唤醒度 {state['arousal']:.2f}
工作记忆中的紧急需求: {wm.get('urgent_need', 'none')}
相关记忆: {wm.get('relevant_memory', '无')}

可选的行动（motor_plan）：
- sit_down, stand_up, look_at_board, sleep, wake_up, relax, move_to_seat, eat_snack, drink_water
同时可以发送自主神经指令（autonomic_commands），例如 [{{"system":"sympathetic","level":0.0-1.0}}]。
请输出 JSON: {{"motor_plan": "...", "autonomic_commands": [...]}}
注意：如果饥饿或口渴很高，优先考虑满足需求。"""
    try:
        resp = llm.call_reasoner(prompt, max_tokens=200)
        data = extract_json_from_llm_response(resp)
        motor_plan = data.get("motor_plan", "relax")
        # 确保 motor_plan 是字符串
        if not isinstance(motor_plan, str):
            motor_plan = "relax"
        auto_cmds = data.get("autonomic_commands", [])
        if not isinstance(auto_cmds, list):
            auto_cmds = []
        return {
            "motor_plan": motor_plan,
            "autonomic_commands": auto_cmds
        }
    except Exception as e:
        print(f"[Prefrontal] parse error: {e}")
        return {"motor_plan": "relax", "autonomic_commands": []}

def motor_cortex_node(state: NervousSystemState) -> dict:
    plan = state["motor_plan"]
    cmds = []
    if plan == "sit_down":
        cmds.append({"command": "sit_down"})
    elif plan == "stand_up":
        cmds.append({"command": "stand_up"})
    elif plan == "look_at_board":
        cmds.append({"command": "look_at_board"})
    elif plan == "sleep":
        cmds.append({"command": "sleep"})
    elif plan == "wake_up":
        cmds.append({"command": "wake_up"})
    elif plan == "move_to_seat":
        cmds.append({"command": "move_to", "target": [1.5, -0.5]})
    elif plan == "eat_snack":
        cmds.append({"command": "eat_snack"})
    elif plan == "drink_water":
        cmds.append({"command": "drink_water"})
    else:
        cmds.append({"command": "relax"})
    return {"motor_commands": cmds}

def cerebellum_node(state: NervousSystemState) -> dict:
    return {}

def autonomic_nervous_node(state: NervousSystemState) -> dict:
    arousal = float(state.get("arousal", 0))
    plan = state.get("motor_plan", "relax")
    symp_level = 0.1 + 0.3 * arousal
    if plan in ("stand_up", "wake_up", "run"):
        symp_level += 0.2
    if state["homeostatic_state"].get("hunger", 0) > 0.8:
        symp_level += 0.1
    symp_level = min(1.0, symp_level)
    auto_cmds = state.get("autonomic_commands", [])
    if not auto_cmds:
        auto_cmds = [{"system": "sympathetic", "level": symp_level}]
    else:
        # 确保 level 是 float
        for cmd in auto_cmds:
            if "level" in cmd:
                cmd["level"] = float(cmd["level"])
    return {"autonomic_commands": auto_cmds}

def language_node(state: NervousSystemState) -> dict:
    pending_msgs = state["working_memory"].get("pending_qq_msgs", [])
    if not pending_msgs:
        return {"language_output": ""}
    auditory = state["sensory_buffer"].get("auditory")
    audio_text = auditory.text_description if auditory else ""
    hs = state["homeostatic_state"]
    prompt = f"""{CHARACTER_PROFILE}
教室情况：{audio_text}
你的动作：{state['motor_plan']}
饥饿：{hs.get('hunger', 0):.2f}
疲劳：{hs.get('muscle_fatigue', 0):.2f}
对讲机消息：{chr(10).join(pending_msgs[-3:])}
请用小鸟游星野的口吻说一句话（1-2句），慵懒语气带“~”。直接输出对话文本，不要加引号。"""
    try:
        resp = llm.call_language(prompt, max_tokens=150)
        resp = resp.strip().strip('"')
        if not resp:
            resp = "ん... 好麻烦啊~"
        return {"language_output": resp}
    except:
        return {"language_output": "唔... 不想动~"}

# ==================== 构建 LangGraph 工作流 ====================
def build_brain_graph():
    builder = StateGraph(NervousSystemState)
    nodes = [
        ("spinal_reflex", spinal_reflex_node),
        ("thalamus", thalamus_relay_node),
        ("hippocampus", hippocampus_memory_node),
        ("amygdala", amygdala_emotion_node),
        ("hypothalamus", hypothalamus_node),
        ("prefrontal", prefrontal_cortex_node),
        ("motor_cortex", motor_cortex_node),
        ("cerebellum", cerebellum_node),
        ("autonomic", autonomic_nervous_node),
        ("language", language_node),
    ]
    for name, fn in nodes:
        builder.add_node(name, fn)

    builder.set_entry_point("spinal_reflex")
    builder.add_conditional_edges(
        "spinal_reflex",
        lambda s: "motor_cortex" if s.get("reflex_trigger") else "thalamus",
        {"motor_cortex": "motor_cortex", "thalamus": "thalamus"}
    )
    builder.add_edge("thalamus", "hippocampus")
    builder.add_edge("hippocampus", "amygdala")
    builder.add_edge("amygdala", "hypothalamus")
    builder.add_edge("hypothalamus", "prefrontal")
    builder.add_edge("prefrontal", "motor_cortex")
    builder.add_edge("motor_cortex", "cerebellum")
    builder.add_edge("cerebellum", "autonomic")
    builder.add_edge("autonomic", "language")
    builder.add_edge("language", END)
    return builder.compile()

# ==================== 状态持久化 ====================
SAVE_FILE = "sunaookami_state.pkl"

def save_system_state(step: int, env: ClassroomEnv, homeo: HomeostaticSimulator,
                      brain_state: NervousSystemState):
    data = {
        "step": step,
        "env": {
            "student_pos": env.student_pos,
            "student_state": env.student_state,
            "head_yaw": env.head_yaw,
            "lighting": env.lighting,
        },
        "homeostatic": asdict(homeo.state),
        "brain_state": {
            "sensory_buffer": {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v for k, v in brain_state.get("sensory_buffer", {}).items()},
            "episodic_memory": brain_state.get("episodic_memory", [])[-30:],
            "working_memory": brain_state.get("working_memory", {}),
            "emotional_valence": brain_state.get("emotional_valence", 0),
            "arousal": brain_state.get("arousal", 0),
            "motor_plan": brain_state.get("motor_plan", ""),
            "attention_focus": brain_state.get("attention_focus", ""),
        },
        "timestamp": time.time()
    }
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"[持久化] 已保存状态 (step={step})")

def load_system_state():
    if not os.path.exists(SAVE_FILE):
        return None, None, None, None
    try:
        with open(SAVE_FILE, "rb") as f:
            data = pickle.load(f)
        env = ClassroomEnv()
        env.student_pos = data["env"]["student_pos"]
        env.student_state = data["env"]["student_state"]
        env.head_yaw = data["env"]["head_yaw"]
        env.lighting = data["env"]["lighting"]
        homeo = HomeostaticSimulator()
        for k, v in data["homeostatic"].items():
            if hasattr(homeo.state, k):
                setattr(homeo.state, k, v)
        brain_partial = data["brain_state"]
        step = data["step"]
        print(f"[持久化] 加载状态成功 (step={step})")
        return step, env, homeo, brain_partial
    except Exception as e:
        print(f"[持久化] 加载失败: {e}")
        return None, None, None, None

# ==================== QQ 桥接 ====================
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
                if data.get("post_type") == "message" and data.get("message_type") == "group":
                    if data.get("group_id") == ONEBOT_GROUP_ID:
                        user = data.get("sender", {}).get("nickname", "?")
                        text = data.get("raw_message", "")
                        print(f"[QQ] {user}: {text}")
                        await current_queue.put({"user": user, "text": text})
            except Exception:
                pass
    except websockets.ConnectionClosed:
        print("[OneBot] 连接断开")
        current_ws = None

# ==================== 主程序 ====================
async def main():
    print("=== Sunaookami Shiroko 生物神经系统模拟系统 ===")
    print("细化版：完整感觉模拟 + 内稳态 ODE + LangGraph 节点 + 持久化 + QQ桥接")

    loaded_step, loaded_env, loaded_homeo, loaded_brain = load_system_state()
    if loaded_env is not None:
        env = loaded_env
        homeo = loaded_homeo
        start_step = loaded_step + 1
        print(f"从 step {loaded_step} 继续运行")
    else:
        env = ClassroomEnv()
        homeo = HomeostaticSimulator(dt=0.5)
        start_step = 0
        loaded_brain = None

    sensory_sim = SensorySimulator(env)
    graph = build_brain_graph()

    if loaded_brain:
        brain_state: NervousSystemState = {
            "sensory_buffer": loaded_brain.get("sensory_buffer", {}),
            "attention_focus": loaded_brain.get("attention_focus", "none"),
            "homeostatic_state": homeo.get_state_dict(),
            "episodic_memory": loaded_brain.get("episodic_memory", []),
            "working_memory": loaded_brain.get("working_memory", {"pending_qq_msgs": []}),
            "emotional_valence": loaded_brain.get("emotional_valence", 0.0),
            "arousal": loaded_brain.get("arousal", 0.0),
            "reflex_trigger": False,
            "motor_plan": loaded_brain.get("motor_plan", "relax"),
            "motor_commands": [],
            "autonomic_commands": [],
            "language_output": "",
            "error": None,
            "timestamp": time.time(),
        }
    else:
        brain_state: NervousSystemState = {
            "sensory_buffer": {},
            "attention_focus": "none",
            "homeostatic_state": homeo.get_state_dict(),
            "episodic_memory": [],
            "working_memory": {"pending_qq_msgs": []},
            "emotional_valence": 0.0,
            "arousal": 0.0,
            "reflex_trigger": False,
            "motor_plan": "relax",
            "motor_commands": [],
            "autonomic_commands": [],
            "language_output": "",
            "error": None,
            "timestamp": time.time(),
        }

    async with serve(handle_connection, WS_HOST, WS_PORT):
        print(f"[Server] 监听 ws://{WS_HOST}:{WS_PORT} 等待 OneBot 连接...")
        step = start_step
        last_save_time = time.time()
        last_status_time = time.time()

        while True:
            # 收集QQ消息
            qq_msgs = []
            while not current_queue.empty():
                try:
                    qq_msgs.append(current_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            wm = brain_state.get("working_memory", {})
            pending = wm.get("pending_qq_msgs", [])
            for m in qq_msgs:
                pending.append(f"{m['user']}: {m['text']}")
            pending = pending[-5:]
            wm["pending_qq_msgs"] = pending
            brain_state["working_memory"] = wm

            # 采集感觉数据
            sensory_data = await sensory_sim.gather_all()
            brain_state["sensory_buffer"] = sensory_data
            brain_state["homeostatic_state"] = homeo.get_state_dict()

            # 运行图
            try:
                result = graph.invoke(brain_state)
                brain_state.update(result)
            except Exception as e:
                print(f"[LangGraph Error] {e}")
                brain_state["error"] = str(e)

            # 执行运动指令
            env.apply_motor_commands(brain_state.get("motor_commands", []))
            for cmd in brain_state.get("motor_commands", []):
                if cmd.get("command") == "eat_snack":
                    homeo.state.glucose += 1.0
                    homeo.state.hunger = max(0, homeo.state.hunger - 0.4)
                elif cmd.get("command") == "drink_water":
                    homeo.state.thirst = max(0, homeo.state.thirst - 0.5)
                    homeo.state.blood_osmolarity -= 2.0

            # 更新内稳态
            homeo.update(
                brain_state.get("motor_commands", []),
                brain_state.get("autonomic_commands", [])
            )

            # 发送QQ消息
            lang_out = brain_state.get("language_output", "")
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
                    wm = brain_state.get("working_memory", {})
                    wm["pending_qq_msgs"] = []
                    brain_state["working_memory"] = wm
                except Exception as e:
                    print(f"[发送失败] {e}")

            # 状态报告
            now = time.time()
            if now - last_status_time >= 30:
                last_status_time = now
                hs = homeo.state
                print("\n========== 系统状态报告 ==========")
                print(f"时间步: {step}")
                print(f"位置: {env.student_pos}, 状态: {env.student_state}")
                tactile = sensory_data.get("tactile")
                tactile_pressure = tactile.pressure_kpa if tactile else 0.0
                print(f"感觉: 视觉焦点={brain_state['attention_focus']}, 触觉压力={tactile_pressure:.1f}")
                print(f"内稳态: 血糖={hs.glucose:.2f}, 饥饿={hs.hunger:.2f}, 疲劳={hs.muscle_fatigue:.2f}, 应激={hs.adrenaline:.2f}")
                print(f"情绪: 效价={brain_state['emotional_valence']:.2f}, 唤醒={brain_state['arousal']:.2f}")
                print(f"运动计划: {brain_state['motor_plan']}")
                print(f"自主神经: {brain_state.get('autonomic_commands', [])}")
                if brain_state.get("error"):
                    print(f"错误: {brain_state['error']}")
                print("==================================\n")

            if now - last_save_time >= 60:
                save_system_state(step, env, homeo, brain_state)
                last_save_time = now

            step += 1
            await asyncio.sleep(2.0)

if __name__ == "__main__":
    asyncio.run(main())
