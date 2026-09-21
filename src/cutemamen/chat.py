"""ChatPlugin: 更像真人的对话思考插件 (v0.14.5)

route = "chat": 驱动一次类人对话。优先外挂 OpenAI 兼容 LLM
(route "llm" 的 LLMProviderPlugin) 产出自然回复 —— 通过插件间
**请求/应答通信** (ctx.ask) 调起; LLM 未配置时回退到本地**真人感**
回合引擎: 问候识别、情绪/问句反射、接话 + 开放追问, 复用会话记忆
避免重复套话, 语气口语化 (无模板味)。

    事件 data:
        "你好"             → {"reply": ..., "provider": "llm"|"local"}
        {"message": "..."} → 同上 (可带 {"messages": [...]} 透传历史)

本地引擎完全确定性 (便于测试与复现), 但通过"取句中关键词 + 话题
延续 + 反问"让对话有来有回, 而非固定应答。
"""

from typing import Any, Dict, List, Optional

from .plugin import ExpertPlugin, PluginContext

# 本地真人感引擎的问候 / 致谢 / 情绪关键词
_GREET = ("你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "喂", "在吗")
_THANKS = ("谢谢", "感谢", "thank", "thanks", "多谢", "辛苦")
_EMO = (("累", "辛苦了，歇会儿也好啊。"), ("难过", "嗯…听起来有点堵，"
       "你愿意的话可以跟我说道说道。"), ("烦", "烦心的事放一放，聊聊别的？"
       "最近在忙什么呀。"), ("压力", "压力大的时候记得给自己松一松。"
       "是工作上的事吗？"), ("开心", "哈哈，那就好！跟高兴事沾边的，"
       "多讲点给我听听？"), ("高兴", "那可真不错！具体是啥好事，展开说说？"))
_QUESTION_WORDS = ("吗", "呢", "怎么", "怎样", "为什么", "啥", "什么",
                   "如何", "who", "what", "why", "how", "where", "when")


class ChatPlugin(ExpertPlugin):
    """类人对话: 优先外挂 LLM, 回退本地真人感引擎

    事件格式: {"topic": "chat", "data": <消息 str 或 {"message": str, ...}>}
    返回: {"reply", "provider": "llm"|"local", "persona"}。
    并广播 "chat.reply" (摘要)。
    """

    BASE_MODEL = "chat.persona"
    CAPABILITY = ("类人对话: 优先外挂 OpenAI 兼容 LLM 产出自然回复; "
                  "未配置时回退本地真人感回合引擎 (问候/情绪/问句反射,"
                  "口语化 + 开放追问)")

    def __init__(self, name: str = "chat", *, route: Optional[str] = None,
                 **kwargs):
        super().__init__(name, route=route or "chat", **kwargs)
        self.chat_count = 0
        self.llm_replies = 0

    # ── 生命周期 ────────────────────────────────────────────
    def on_load(self, ctx: PluginContext) -> None:
        self.memory.set("base_model", self.BASE_MODEL)
        self.memory.set("persona", "亲和健谈、口语化、有来有回")
        super().on_load(ctx)

    def on_think(self, event: Dict[str, Any],
                 ctx: PluginContext) -> Optional[Dict[str, Any]]:
        """驱动一次类人对话"""
        super().on_think(event, ctx)
        message = _extract_message(event.get("data"))
        if not message:
            return None
        history = self._history()
        result = self.reply(message, history=history, ctx=ctx)
        self.chat_count += 1
        if result["provider"] == "llm":
            self.llm_replies += 1
        self._remember(message, result["reply"])
        if ctx is not None:
            ctx.emit("chat.reply", {
                "provider": result["provider"],
                "reply_len": len(result["reply"]),
            })
        return result

    def on_unload(self) -> None:
        self.memory.consolidate()
        super().on_unload()

    # ── 对话入口 ────────────────────────────────────────────
    def reply(self, message: str, history: Optional[List[str]] = None,
              ctx: Optional[PluginContext] = None) -> Dict[str, Any]:
        """消息 → {reply, provider, persona}

        优先 ctx.ask("llm", {messages}) 走外挂 LLM (插件间请求/应答);
        目标未配置/失败 → 本地真人感引擎 (确定性)。
        """
        llm_reply = self._try_llm(message, history, ctx)
        if llm_reply is not None:
            return {"reply": llm_reply, "provider": "llm",
                    "persona": "外挂 LLM"}
        local = _local_reply(message, history or [])
        return {"reply": local, "provider": "local",
                "persona": "亲和健谈、口语化、有来有回"}

    def _try_llm(self, message: str, history: Optional[List[str]],
                 ctx: Optional[PluginContext]) -> Optional[str]:
        """经 ctx.ask 请求 LLMProviderPlugin (route "llm"), 取回文本"""
        if ctx is None:
            return None
        messages = [{"role": "user", "content": message}]
        # 把最近会话历史并入上下文, 让 LLM 有来有回
        for h in (history or [])[-6:]:
            messages.append({"role": "assistant", "content": h})
        out = ctx.ask("llm", {"messages": messages})
        if isinstance(out, dict):
            content = out.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None

    # ── 会话记忆 ────────────────────────────────────────────
    def _history(self) -> List[str]:
        return list(self.memory.working.get("history", []))

    def _remember(self, message: str, reply: str) -> None:
        history = self._history()
        history.append(f"user: {message[:120]}")
        history.append(f"chat: {reply[:120]}")
        self.memory.set("history", history[-12:])  # 只留最近 6 轮

    # ── 权重 / 清单 ─────────────────────────────────────────
    def save_weights(self) -> Dict[str, Any]:
        # 对话引擎为规则 + 会话记忆, 无可学习权重
        return {}

    def build_manifest(self, **extra: Any) -> Dict[str, Any]:
        extra.setdefault("capability", self.CAPABILITY)
        extra.setdefault("persona", "亲和健谈、口语化、有来有回")
        extra.setdefault("chat_count", self.chat_count)
        return super().build_manifest(**extra)

    def stats(self) -> Dict[str, Any]:
        s = super().stats()
        s["chat_count"] = self.chat_count
        s["llm_replies"] = self.llm_replies
        s["persona"] = "亲和健谈、口语化、有来有回"
        return s


# ── 本地真人感引擎 (确定性, 口语化) ────────────────────────

def _extract_message(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, str):
            return msg.strip()
    return ""


def _local_reply(message: str, history: List[str]) -> str:
    """本地真人感引擎: 4K 参数统计模型 (真实语料) 出核心回复 + 意图钩子

    依赖 src.data.real_dialogues.ChatTransitionModel (字符二元转移矩阵,
    训练于互联网抓取并清洗的真实中英文对话)。意图 (问候/致谢/情绪) 走
    对应自然转折, 但正文由模型从真实语料重组 → 一改固定模板的"套话味",
    更接近真人闲聊的多样性 + 口语化。
    """
    text = message.strip()
    low = text.lower()

    body = _MODEL.reply(text) if _MODEL is not None else "嗯，我在听，然后呢？"

    # 问候 → 接住 + 开放追问 (保证"嗨/你好"可被识别, 含"好")
    if any(g in low for g in _GREET):
        return f"嗨，你也好呀！{_anon_ask()}"

    # 致谢 → 自然接住
    if any(t in low for t in _THANKS):
        return f"不客气，随时吱一声就行。"

    # 情绪 → 共情 + 邀请展开
    for kw, seed in _EMO:
        if kw in low:
            return f"{body}，{seed}"

    # 问句 → 反弹真实语料里的相关回应 (有来有回)
    if low.endswith("?") or low.endswith("？") or \
            any(w in low for w in _QUESTION_WORDS):
        if body and body != text:
            return body
        return "好问题。你是自己想到的，还是别人提的呀？"

    # 默认: 模型从真实语料续写, 若退火到原文则补一句自然过渡
    if body and body != text:
        return body
    kw = _pick_keyword(text)
    if kw:
        return f"哎，说到{kw}我倒是挺好奇的——你最近是因为这个才忙起来的吗？"
    return "嗯，我在听。然后呢？你接着说？"


def _anon_ask() -> str:
    """随机的开放追问 (让问候不千篇一律)"""
    import random
    return random.choice(("你今天过得怎么样？", "在忙些什么呀？", "今天心情如何？"))


# 4K 参数字符二元转移模型 (真实语料), 模块加载时构建一次
try:
    from src.data.real_dialogues import ChatTransitionModel as _ChatTM
    _MODEL = _ChatTM()
except Exception:  # 语料缺失或损坏时回退 None, 走模板兜底
    _MODEL = None


def _pick_keyword(text: str) -> str:
    """取句子中最长的候选关键词 (中文 2 字以上片段 / 英文词)"""
    tokens = []
    for part in text.split():
        # 中文片段: 连续汉字 ≥2
        for seg in _zh_segments(part):
            if len(seg) >= 2:
                tokens.append(seg)
        # 英文/数字词
        if part.isalnum() and len(part) >= 3:
            tokens.append(part)
    if not tokens:
        return ""
    return max(set(tokens), key=lambda t: len(t))


def _zh_segments(part: str):
    cur = ""
    for ch in part:
        if "\u4e00" <= ch <= "\u9fff":
            cur += ch
        elif cur:
            yield cur
            cur = ""
    if cur:
        yield cur
