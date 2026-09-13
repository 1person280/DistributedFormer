"""
CubeGPT 终端聊天 (v0.7.1)

简单的 REPL: 用户输入喂进 CubeGPT 的 text 面, 跑若干步后由输出模块的
脉冲模式生成回复。CubeGPT 目前是脉冲 reservoir (没有语言生成头),
回复由脉冲统计特征 (脉冲数 / 平均强度 / 主活跃面) 驱动模板生成,
并如实展示网络内部状态。对话过程 STDP 在线学习, 输入模式会被记住。

用法:
    dformer chat            # 默认深度 2
    dformer chat --depth 1  # 轻量深度
交互命令:
    /reset  清空网络与 KV 记忆, 开启新对话
    /stats  查看网络统计 (各面脉冲 / KV 用量 / STDP)
    /exit   退出 (Ctrl+D / Ctrl+C 同效)
"""

import sys
import time

import numpy as np

from distributedformer.core.distributedformer import CubeGPT

_BANNER = r"""
════════════════════════════════════════════════════
  CubeGPT 终端聊天 v{version}
  {scale}
  (脉冲 reservoir, 回复由输出脉冲模式驱动; /help 查看命令)
════════════════════════════════════════════════════"""

_HELP = """可用命令:
  /reset   清空网络状态与 KV 记忆, 开启新对话
  /stats   显示网络统计 (各面脉冲 / KV 堆 / STDP 学习)
  /help    显示本帮助
  /exit    退出聊天 (Ctrl+D 同效)"""

_STEPS_PER_TURN = 8


def _response_from_spikes(gpt: CubeGPT, text: str) -> str:
    """根据本回合网络活动挑一句模板回复。

    CubeGPT 没有语言生成头, 这里如实把网络活动 (输出头部模式能量 +
    脉冲统计) 映射为固定语气的回应, 并附上内部状态, 让用户看到
    "它在想什么"。长输入 / 新奇词汇会点燃更多单元。
    """
    spikes = gpt.output_module.last_spikes
    n_spikes = len(spikes)
    strength = (
        sum(s.payload.strength for s in spikes) / n_spikes if spikes else 0.0
    )
    pattern = gpt.get_output_pattern()
    energy = float(np.linalg.norm(pattern))
    face_counts = {
        m: len(gpt.faces[m].last_spikes) for m in gpt.ring if m in gpt.faces
    }
    face_total = sum(face_counts.values())
    active_face = max(face_counts, key=face_counts.get) if face_total else None
    kv_used = len(gpt.kv_stack.entries)

    if n_spikes >= 6 or energy > 3.0:
        reply = "强烈共鸣! 输入把网络点燃了, 输出模式能量很高。"
    elif n_spikes > 0:
        reply = "收到, 输出头部发射了几枚脉冲, 模式清晰。"
    elif face_total >= 8:
        reply = "皮层各面活动踊跃, 但还没传导到输出头部, 继续多聊几句看积累效果。"
    elif face_total > 0:
        reply = "有一些脉冲活动, 但还不算强, 网络节律正处抑制相。"
    else:
        reply = "这一拍很安静 (0 脉冲)。输入模式太弱, 换个说法再试试。"

    detail = f"[输出脉冲 {n_spikes} | 模式能量 {energy:.2f}"
    if active_face:
        detail += f" | 活跃面 {active_face}"
    detail += f" | KV {kv_used}]"
    return f"{reply}\n  {detail}"


class CubeGPTChat:
    """封装一个 CubeGPT 实例与对话回合逻辑 (便于测试)。"""

    def __init__(self, depth: int = 2, dim: int = 16):
        self.gpt = CubeGPT(depth=depth, dim=dim, training_mode=True)

    def reply(self, text: str) -> str:
        """一个对话回合: 文本喂 text 面 → 多步运行 → 脉冲统计回复。"""
        for _ in range(_STEPS_PER_TURN):
            self.gpt.step({"text": text})
        return _response_from_spikes(self.gpt, text)

    def reset(self) -> None:
        self.gpt.reset_state()
        self.gpt.kv_stack.entries.clear()

    def stats(self) -> str:
        scale = self.gpt.faces
        lines = ["[网络]"]
        for m, face in self.gpt.faces.items():
            lines.append(
                f"  {m:<11} 单元 {len(face.get_units())}, "
                f"本回合脉冲 {len(face.last_spikes)}"
            )
        lines.append(f"[KV 堆] 用量 {len(self.gpt.kv_stack.entries)}")
        stdp = self.gpt.get_stdp_stats()
        lines.append(
            f"[STDP] LTP {stdp['total_ltp']} / LTD {stdp['total_ltd']}, "
            f"累计权重变化 {stdp['total_weight_change']:.3f}"
        )
        lines.append(f"[总步数] {self.gpt.total_steps}")
        return "\n".join(lines)

    @property
    def scale_line(self) -> str:
        from distributedformer.core.distributedformer import calculate_cube_scale
        return calculate_cube_scale(depth=self.gpt.depth,
                                    n_faces=len(self.gpt.faces))["description"]


def run_chat(depth: int = 2, dim: int = 16) -> None:
    from distributedformer import __version__

    chat = CubeGPTChat(depth=depth, dim=dim)
    print(_BANNER.format(version=__version__, scale=chat.scale_line))
    print(_HELP)

    while True:
        try:
            text = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] 再见!")
            return
        if not text:
            continue
        if text.startswith("/"):
            cmd = text.lower()
            if cmd in ("/exit", "/quit"):
                print("[chat] 再见!")
                return
            if cmd == "/reset":
                chat.reset()
                print("[chat] 网络状态与 KV 记忆已清空, 新对话开始。")
            elif cmd == "/stats":
                print(chat.stats())
            elif cmd == "/help":
                print(_HELP)
            else:
                print(f"[chat] 未知命令 {text} ({_HELP.splitlines()[0]})")
            continue
        started = time.time()
        print(f"CubeGPT > {chat.reply(text)}")
        print(f"  ({time.time() - started:.2f}s, 共 {chat.gpt.total_steps} 步)")


if __name__ == "__main__":
    run_chat()
