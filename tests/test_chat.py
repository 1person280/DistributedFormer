"""v0.7.1 终端聊天: CubeGPTChat 回合逻辑测试"""

from distributedformer.demos.chat import CubeGPTChat


def test_reply_is_str_with_state_detail():
    chat = CubeGPTChat(depth=1)
    out = chat.reply("hello world")
    assert isinstance(out, str) and "输出脉冲" in out


def test_conversation_accumulates_steps():
    chat = CubeGPTChat(depth=1)
    steps0 = chat.gpt.total_steps
    chat.reply("first")
    chat.reply("second")
    assert chat.gpt.total_steps > steps0


def test_reset_clears_state():
    chat = CubeGPTChat(depth=1)
    chat.reply("something")
    assert chat.gpt.total_steps > 0
    chat.reset()
    assert chat.gpt.total_steps == 0
    assert len(chat.gpt.kv_stack.entries) == 0


def test_stats_reports_faces_and_kv():
    chat = CubeGPTChat(depth=1)
    stats = chat.stats()
    for m in ("numeric", "text", "timeseries", "image"):
        assert m in stats
    assert "KV" in stats and "STDP" in stats
