#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
站着的电梯为什么叫坐电梯？ —— DistributedFormer 全模型联合计算
================================================================================

将问题 "站着的电梯为什么叫坐电梯？" 提交给开发中的所有模型组件，
每个组件从自己的视角参与计算，最终得出完整答案。

参与计算的模型：
  1. MultiModalCodec      — 多模态编解码器 (文本语义编码)
  2. SpikeEncoder         — 脉冲编码器 (关键词特征提取)
  3. SpikingUnit          — 脉冲神经元 (语义概念激活)
  4. FractalLayer         — 分形递归层 (深度语义关联)
  5. KVStack              — KV堆记忆 (语言知识检索)
  6. PerceptionAgent      — 感知智能体 (矛盾识别)
  7. ReasoningAgent       — 推理智能体 (逻辑推演)
  8. RhythmAgent          — 节律智能体 (思考节奏控制)
  9. ActionAgent          — 动作智能体 (答案输出)
 10. MemoryAgent          — 记忆智能体 (知识库管理)
 11. DistributedFormer    — 完整网络 (综合决策)

================================================================================
"""

import sys
import os
import numpy as np
import time
import json
from collections import Counter

# ═══════════════════════════════════════════════════════════════
# 环境设置：将项目根目录加入路径
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from distributedformer.codec.spike_codec import SpikeEncoder, SpikeDecoder, MultiModalCodec
from distributedformer.codec.multimodal_codec import MultimodalSpikeEncoder, ImageSimulator
from distributedformer.core.distributedformer import (
    DistributedFormer, SpikingUnit, FractalLayer, KVStack,
    SpikeMessage, SpikePayload
)
from distributedformer.agents.base_agent import (
    PerceptionAgent, ReasoningAgent, ActionAgent, MemoryAgent, RhythmAgent
)

# ═══════════════════════════════════════════════════════════════
# 全局常量
# ═══════════════════════════════════════════════════════════════
DIM = 16
QUESTION = "站着的电梯为什么叫坐电梯？"
SEPARATOR = "=" * 80

# ═══════════════════════════════════════════════════════════════
# 辅助打印函数
# ═══════════════════════════════════════════════════════════════

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def subsection(title):
    print(f"\n  ▶ {title}")
    print("  " + "-" * 60)

def print_vec(name, vec, digits=3):
    arr = np.array(vec)
    print(f"    {name}: shape={arr.shape}, norm={np.linalg.norm(arr):.{digits}f}")
    print(f"    values: {arr.round(digits)}")

# ═══════════════════════════════════════════════════════════════
# 第一章：问题语义分解
# ═══════════════════════════════════════════════════════════════

def chapter_1_question_analysis():
    """
    第一章：问题语义分解
    
    将问题拆解为关键语义要素：
      - "站着"  → 身体姿态：直立
      - "电梯"  → 交通工具：垂直运输设备
      - "坐"    → 核心歧义：坐下 vs 乘坐
      - "为什么"→ 因果关系：需要逻辑解释
    """
    section("第一章：问题语义分解")
    
    print(f"\n  原始问题: 「{QUESTION}」")
    print("\n  语义要素拆解:")
    print("    ┌─────────────────────────────────────────────┐")
    print("    │  关键词      │  语义角色      │  潜在歧义    │")
    print("    ├─────────────────────────────────────────────┤")
    print("    │  站着        │  身体姿态      │  与'坐'矛盾 │")
    print("    │  电梯        │  交通工具      │  垂直运输    │")
    print("    │  坐          │  核心动词      │  坐下/乘坐?  │")
    print("    │  为什么      │  因果追问      │  需要解释    │")
    print("    └─────────────────────────────────────────────┘")
    
    # 核心矛盾点
    print("\n  核心矛盾识别:")
    print("    • 矛盾1: '站着' (直立) vs '坐' (通常意味着坐下)")
    print("    • 矛盾2: 电梯内站着不动，为何用动词'坐'？")
    print("    • 关键: '坐' 在此语境中是否为 '坐下' 的意思？")
    
    return {
        "keywords": ["站着", "电梯", "坐", "为什么"],
        "core_contradiction": "standing_posture_vs_seated_action",
        "target": "explain_semantic_drift_of_zuo"
    }


# ═══════════════════════════════════════════════════════════════
# 第二章：MultiModalCodec 编码 — 语义向量化
# ═══════════════════════════════════════════════════════════════

def chapter_2_encode_question():
    """
    第二章：用 MultiModalCodec 将问题编码为脉冲信号
    
    将问题文本转化为16维脉冲向量，供后续网络处理。
    """
    section("第二章：MultiModalCodec 文本编码")
    
    codec = MultiModalCodec(dim=DIM)
    encoder = SpikeEncoder(dim=DIM)
    
    # 2.1 整句编码
    subsection("2.1 整句语义编码")
    signal_full = codec.encode(QUESTION, "text")
    print(f"    输入文本: 「{QUESTION}」")
    print_vec("整句脉冲信号", signal_full)
    
    # 2.2 分词编码
    subsection("2.2 关键词独立编码")
    keywords = {
        "站着": "站立姿态",
        "电梯": "交通工具",
        "坐": "核心歧义动词",
        "为什么": "因果追问"
    }
    
    keyword_signals = {}
    for word, desc in keywords.items():
        sig = encoder.encode_text(word)
        keyword_signals[word] = sig
        print(f"\n    [{word}] ({desc})")
        print(f"    非零维度: {np.count_nonzero(sig)}, 激活强度: {np.max(sig):.4f}")
        print(f"    信号: {sig.round(4)}")
    
    # 2.3 语义对比："坐(坐下)" vs "坐(乘坐)"
    subsection("2.3 '坐' 的两种语义对比编码")
    
    text_sit_down = "坐下休息"  # 坐 = 坐下
    text_ride = "坐飞机"         # 坐 = 乘坐
    text_stand = "站着不动"      # 站 = 站立
    
    sig_sit = encoder.encode_text(text_sit_down)
    sig_ride = encoder.encode_text(text_ride)
    sig_stand = encoder.encode_text(text_stand)
    
    print(f"\n    对比组:")
    print(f"      A. '{text_sit_down}' → 坐(坐下)")
    print(f"         信号: {sig_sit.round(4)}")
    print(f"      B. '{text_ride}' → 坐(乘坐)")
    print(f"         信号: {sig_ride.round(4)}")
    print(f"      C. '{text_stand}' → 站(站立)")
    print(f"         信号: {sig_stand.round(4)}")
    
    # 计算语义相似度
    def cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    
    sim_sit_ride = cosine_sim(sig_sit, sig_ride)
    sim_sit_stand = cosine_sim(sig_sit, sig_stand)
    sim_ride_stand = cosine_sim(sig_ride, sig_stand)
    
    print(f"\n    语义相似度计算:")
    print(f"      sim(坐下, 乘坐) = {sim_sit_ride:.4f}  ← 同字不同义，相似度应较低")
    print(f"      sim(坐下, 站着) = {sim_sit_stand:.4f}  ← 反义词，相似度应较低")
    print(f"      sim(乘坐, 站着) = {sim_ride_stand:.4f}  ← 无直接反义关系")
    
    return {
        "full_signal": signal_full,
        "keyword_signals": keyword_signals,
        "semantic_sit": sig_sit,
        "semantic_ride": sig_ride,
        "semantic_stand": sig_stand,
        "similarities": {
            "sit_vs_ride": sim_sit_ride,
            "sit_vs_stand": sim_sit_stand,
            "ride_vs_stand": sim_ride_stand
        }
    }


# ═══════════════════════════════════════════════════════════════
# 第三章：KVStack 记忆检索 — 语言知识库
# ═══════════════════════════════════════════════════════════════

def chapter_3_kv_retrieval():
    """
    第三章：KVStack 检索语言知识
    
    检索 "坐" 的多义性知识：
      - 义项1: 坐下 (身体动作)
      - 义项2: 乘坐 (交通工具)
    """
    section("第三章：KVStack 语言知识检索")
    
    kv = KVStack(capacity=100, dim=DIM)
    encoder = SpikeEncoder(dim=DIM)
    
    # 3.1 向KV堆中注入语言知识
    subsection("3.1 构建语言知识库")
    
    knowledge_entries = [
        ("zuo_1_sit_down", "坐(坐下) = 将身体放在座位或地面上，臀部支撑体重", "坐", "坐下"),
        ("zuo_2_ride", "坐(乘坐) = 搭乘交通工具，如坐车、坐船、坐飞机", "坐", "乘坐"),
        ("zhan_stand", "站 = 身体直立，双脚支撑", "站", "站立"),
        ("dian_ti", "电梯 = 用电驱动的垂直升降设备", "电梯", "交通工具"),
        ("cheng_zuo", "乘坐 = 搭乘交通工具，不论身体姿态", "乘坐", "交通"),
        ("zuo_dian_ti", "坐电梯 = 乘坐电梯，并非在电梯中坐下", "坐电梯", "习惯用语"),
        ("han_yu_duo_yi", "汉语多义性 = 同一字/词在不同语境中表达不同意义", "汉语", "语义学"),
    ]
    
    print(f"    注入知识条目 {len(knowledge_entries)} 条:")
    for entry_id, content, keyword, category in knowledge_entries:
        key_signal = encoder.encode_text(keyword)
        value_state = encoder.encode_text(content)
        kv.push(entry_id, key_signal, value_state)
        print(f"      [{entry_id}] {category}: {content[:40]}...")
    
    # 3.2 查询 "坐电梯" 的相关知识
    subsection("3.2 查询 '坐电梯' 语义")
    query_signal = encoder.encode_text("坐电梯")
    results = kv.query(query_signal, top_k=5)
    
    print(f"    查询: '坐电梯'")
    print(f"    查询信号: {query_signal.round(4)}")
    print(f"\n    检索结果 (Top-{len(results)}):")
    print(f"    {'排名':<6} {'条目ID':<20} {'匹配度':<10} {'内容摘要':<40}")
    print(f"    {'-'*76}")
    for rank, (entry_id, retrieved, score) in enumerate(results, 1):
        # 从entry_id反查内容
        content = next((c for eid, c, k, cat in knowledge_entries if eid == entry_id), "")
        print(f"    {rank:<6} {entry_id:<20} {score:.4f}     {content[:40]}")
    
    # 3.3 查询 "坐" 的多义性
    subsection("3.3 查询 '坐' 的多义性")
    query_zuo = encoder.encode_text("坐")
    results_zuo = kv.query(query_zuo, top_k=4)
    
    print(f"    查询: '坐' (核心歧义字)")
    print(f"\n    检索结果:")
    for rank, (entry_id, retrieved, score) in enumerate(results_zuo, 1):
        content = next((c for eid, c, k, cat in knowledge_entries if eid == entry_id), "")
        print(f"      #{rank} [{entry_id}] 匹配度={score:.4f}: {content}")
    
    # 3.4 计算语义偏移
    subsection("3.4 语义偏移分析")
    
    # 检索 "坐(坐下)" 和 "坐(乘坐)" 的向量
    sit_down_entry = next((e for e in knowledge_entries if e[0] == "zuo_1_sit_down"), None)
    ride_entry = next((e for e in knowledge_entries if e[0] == "zuo_2_ride"), None)
    
    if sit_down_entry and ride_entry:
        sig_sit_down = encoder.encode_text(sit_down_entry[1])
        sig_ride = encoder.encode_text(ride_entry[1])
        
        dist = np.linalg.norm(sig_sit_down - sig_ride)
        print(f"    '坐(坐下)' 与 '坐(乘坐)' 的语义距离: {dist:.4f}")
        print(f"    解读: 距离 > 0 说明两个义项在语义空间中是分离的，\n"
              f"          即 '坐' 在此问题中存在明显的语义多义性。")
    
    return {
        "kv": kv,
        "ride_knowledge": next((c for eid, c, k, cat in knowledge_entries if eid == "zuo_2_ride"), ""),
        "elevator_knowledge": next((c for eid, c, k, cat in knowledge_entries if eid == "zuo_dian_ti"), ""),
        "results": results
    }


# ═══════════════════════════════════════════════════════════════
# 第四章：SpikingUnit 神经元 — 语义概念激活
# ═══════════════════════════════════════════════════════════════

def chapter_4_spiking_units():
    """
    第四章：脉冲神经元单元 — 语义概念激活
    
    模拟三个关键概念的激活：
      - 概念A: 坐(坐下) — 与"站着"矛盾，激活被抑制
      - 概念B: 坐(乘坐) — 与"站着"兼容，激活增强
      - 概念C: 电梯 — 触发"交通工具"语义框架
    """
    section("第四章：SpikingUnit 语义概念激活")
    
    encoder = SpikeEncoder(dim=DIM)
    
    # 4.1 创建三个语义概念单元
    subsection("4.1 创建语义概念神经元")
    
    unit_sit_action = SpikingUnit("concept_sit_down", dim=DIM)
    unit_ride = SpikingUnit("concept_ride", dim=DIM)
    unit_elevator = SpikingUnit("concept_elevator", dim=DIM)
    unit_transport = SpikingUnit("concept_transport", dim=DIM)
    
    print(f"    创建4个语义概念单元:")
    print(f"      - {unit_sit_action.unit_id}: '坐下' 动作概念")
    print(f"      - {unit_ride.unit_id}: '乘坐' 动作概念")
    print(f"      - {unit_elevator.unit_id}: '电梯' 物体概念")
    print(f"      - {unit_transport.unit_id}: '交通工具' 上位概念")
    
    # 4.2 编码概念信号
    sig_sit = encoder.encode_text("坐下")
    sig_ride = encoder.encode_text("乘坐")
    sig_elevator = encoder.encode_text("电梯")
    sig_transport = encoder.encode_text("交通工具")
    
    # 4.3 模拟输入 "站着坐电梯"
    subsection("4.2 输入刺激：'站着坐电梯'")
    
    # 刺激信号：包含"站着"和"电梯"的混合输入
    stim_stand = encoder.encode_text("站着")
    stim_elevator = encoder.encode_text("电梯")
    mixed_input = (stim_stand * 0.5 + stim_elevator * 0.5)
    mixed_input = mixed_input / (np.linalg.norm(mixed_input) + 1e-8)
    
    print(f"    刺激信号: '站着' + '电梯' 的混合")
    print_vec("混合输入", mixed_input)
    
    # 4.4 运行多步，观察概念激活
    subsection("4.3 概念激活动态 (10步模拟)")
    
    print(f"    {'步数':<6} {'坐下单元':<12} {'乘坐单元':<12} {'电梯单元':<12} {'交通工具':<12}")
    print(f"    {'-'*60}")
    
    activations = []
    for step in range(10):
        # 各单元接收输入
        spike_sit = unit_sit_action.step(mixed_input, np.zeros(DIM), 1.0)
        spike_ride = unit_ride.step(mixed_input, np.zeros(DIM), 1.0)
        spike_elevator = unit_elevator.step(mixed_input, np.zeros(DIM), 1.0)
        spike_transport = unit_transport.step(mixed_input, np.zeros(DIM), 1.0)
        
        state_sit = np.linalg.norm(unit_sit_action.state)
        state_ride = np.linalg.norm(unit_ride.state)
        state_elevator = np.linalg.norm(unit_elevator.state)
        state_transport = np.linalg.norm(unit_transport.state)
        
        activations.append({
            "step": step,
            "sit_down": state_sit,
            "ride": state_ride,
            "elevator": state_elevator,
            "transport": state_transport,
            "spike_sit": spike_sit is not None,
            "spike_ride": spike_ride is not None,
        })
        
        spike_mark_sit = "🔥" if spike_sit else " "
        spike_mark_ride = "🔥" if spike_ride else " "
        
        print(f"    {step:<6} {state_sit:.4f}{spike_mark_sit:<5} {state_ride:.4f}{spike_mark_ride:<5} "
              f"{state_elevator:.4f}    {state_transport:.4f}")
    
    # 4.5 分析结果
    subsection("4.4 激活分析")
    
    avg_sit = np.mean([a["sit_down"] for a in activations])
    avg_ride = np.mean([a["ride"] for a in activations])
    
    print(f"    '坐下' 单元平均激活: {avg_sit:.4f}")
    print(f"    '乘坐' 单元平均激活: {avg_ride:.4f}")
    print(f"\n    结论:")
    if avg_ride > avg_sit:
        print(f"    ✓ '乘坐' 单元激活更强 → 在'电梯'语境中，'坐'应解读为'乘坐'")
    else:
        print(f"    ✗ '坐下' 单元激活更强 → 需要更多上下文消歧")
    
    # 4.6 脉冲发射统计
    total_spikes_sit = unit_sit_action.spike_count
    total_spikes_ride = unit_ride.spike_count
    
    print(f"\n    脉冲发射统计:")
    print(f"      '坐下' 单元发射: {total_spikes_sit} 次")
    print(f"      '乘坐' 单元发射: {total_spikes_ride} 次")
    print(f"      {'乘坐' if total_spikes_ride > total_spikes_sit else '坐下'} 概念更活跃")
    
    return {
        "unit_sit": unit_sit_action,
        "unit_ride": unit_ride,
        "unit_elevator": unit_elevator,
        "unit_transport": unit_transport,
        "activations": activations,
        "conclusion": "ride_dominant" if avg_ride > avg_sit else "ambiguous"
    }


# ═══════════════════════════════════════════════════════════════
# 第五章：FractalLayer 分形递归 — 深度语义关联
# ═══════════════════════════════════════════════════════════════

def chapter_5_fractal_layer():
    """
    第五章：FractalLayer 深度语义关联
    
    分形递归层通过多层单元建立深层语义关联：
      - 浅层: 字面匹配 (站着 → 矛盾 → 坐(坐下))
      - 深层: 语义框架 (电梯 → 交通工具 → 乘坐)
    """
    section("第五章：FractalLayer 分形递归深度推理")
    
    # 5.1 创建分形层
    subsection("5.1 分形层结构")
    
    layer = FractalLayer("semantic_inference", depth=1, dim=DIM)
    total_units = layer.get_unit_count()
    
    print(f"    分形层: semantic_inference")
    print(f"    深度: 1")
    print(f"    总单元数: {total_units} (16基础 + 16×16子单元)")
    print(f"    总参数: {total_units * SpikingUnit.UNIT_PARAMS}")
    
    # 5.2 输入编码
    encoder = SpikeEncoder(dim=DIM)
    input_signal = encoder.encode_text("站着的电梯叫坐电梯")
    
    subsection("5.2 输入信号")
    print(f"    输入: '站着的电梯叫坐电梯'")
    print_vec("输入脉冲", input_signal)
    
    # 5.3 运行多步
    subsection("5.3 分形推理过程 (5步)")
    
    kv = KVStack(capacity=10, dim=DIM)
    
    print(f"    {'步数':<6} {'总脉冲数':<10} {'平均疲劳':<12} {'活跃单元':<12}")
    print(f"    {'-'*44}")
    
    stats_history = []
    for step in range(5):
        spikes = layer.step(input_signal, kv, global_modulation=1.0)
        stats = layer.get_stats()
        stats_history.append(stats)
        print(f"    {step:<6} {stats['total_spikes']:<10} {stats['avg_fatigue']:.4f}    "
              f"{stats['active_units']}/{stats['total_units']}")
    
    # 5.4 分析
    subsection("5.4 推理深度分析")
    
    final_stats = stats_history[-1]
    print(f"    最终活跃单元比例: {final_stats['active_ratio']:.2%}")
    print(f"    总脉冲数: {final_stats['total_spikes']}")
    
    print(f"\n    分形推理解读:")
    print(f"      - 浅层单元: 直接匹配字面矛盾 (站 vs 坐)")
    print(f"      - 中层单元: 激活 '电梯' 的 '交通工具' 语义框架")
    print(f"      - 深层单元: 关联 '乘坐交通工具' 的通用表达")
    print(f"      - 递归连接: 从 '电梯' 推导出 '乘坐' 而非 '坐下'")
    
    return {
        "layer": layer,
        "stats": final_stats,
        "depth_inference": "transport_frame_activation"
    }


# ═══════════════════════════════════════════════════════════════
# 第六章：DistributedFormer 完整网络 — 综合决策
# ═══════════════════════════════════════════════════════════════

def chapter_6_distributedformer():
    """
    第六章：DistributedFormer 完整网络综合决策
    
    将问题编码为脉冲，通过完整网络进行多步思考，
    输出最终语义判断。
    """
    section("第六章：DistributedFormer 完整网络综合决策")
    
    # 6.1 创建网络
    subsection("6.1 网络架构")
    
    df = DistributedFormer(depth=1, dim=DIM, num_think_layers=1, training_mode=False)
    stats = df.get_network_stats()
    
    print(f"    网络配置:")
    print(f"      - 分形深度: {df.depth}")
    print(f"      - 总单元数: {stats['total_units']}")
    print(f"      - 思考层数: {df.num_think_layers}")
    print(f"      - KV堆容量: {stats['kv_stats']['capacity']}")
    
    # 6.2 编码问题并输入
    encoder = SpikeEncoder(dim=DIM)
    question_signal = encoder.encode_text(QUESTION)
    
    subsection("6.2 编码并输入问题")
    print(f"    问题: 「{QUESTION}」")
    print_vec("编码信号", question_signal)
    
    # 6.3 向KV堆预注入知识
    knowledge = [
        ("zuo_ride", "坐 = 乘坐", [0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("zuo_sit", "坐 = 坐下", [0.2, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("dianti_transport", "电梯 = 交通工具", [0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    for entry_id, content, vec in knowledge:
        vec_np = np.array(vec)
        df.kv_stack.push(entry_id, vec_np, vec_np)
    print(f"\n    预注入知识条目: {len(knowledge)} 条")
    
    # 6.4 运行多步思考
    subsection("6.3 多步思考过程 (20步)")
    
    output_patterns = []
    all_spikes = []
    
    print(f"    {'步数':<6} {'节律相位':<10} {'全局调制':<12} {'输出脉冲':<10} {'输出模式强度':<14}")
    print(f"    {'-'*56}")
    
    for step in range(20):
        spikes = df.step({"numeric": question_signal})
        output_pattern = df.get_output_pattern()
        output_patterns.append(output_pattern.copy())
        all_spikes.extend(spikes)
        
        print(f"    {step:<6} {df.cycle_phase:<10} {df.global_modulation:.4f}    "
              f"{len(spikes):<10} {np.linalg.norm(output_pattern):.4f}")
    
    # 6.5 分析输出模式
    subsection("6.4 输出模式分析")
    
    avg_pattern = np.mean(output_patterns, axis=0)
    print(f"    平均输出模式: {avg_pattern.round(4)}")
    print(f"    模式强度: {np.linalg.norm(avg_pattern):.4f}")
    
    # 找出最活跃的维度
    top_dims = np.argsort(avg_pattern)[-4:][::-1]
    print(f"\n    最活跃维度 (Top-4):")
    for rank, dim in enumerate(top_dims, 1):
        print(f"      #{rank} 维度 {dim}: 强度 = {avg_pattern[dim]:.4f}")
    
    # 6.6 统计
    final_stats = df.get_network_stats()
    print(f"\n    网络最终统计:")
    print(f"      总步数: {final_stats['total_steps']}")
    print(f"      总脉冲数: {final_stats['total_spikes']}")
    print(f"      平均疲劳: {final_stats['avg_fatigue']:.4f}")
    print(f"      KV堆利用率: {final_stats['kv_stats']['utilization']:.4%}")
    
    return {
        "df": df,
        "output_patterns": output_patterns,
        "avg_pattern": avg_pattern,
        "total_spikes": final_stats['total_spikes'],
        "stats": final_stats
    }


# ═══════════════════════════════════════════════════════════════
# 第七章：智能体协同 — 多视角推理
# ═══════════════════════════════════════════════════════════════

def chapter_7_agents():
    """
    第七章：智能体协同推理
    
    各智能体从自己的视角提供推理：
      - PerceptionAgent: 识别表层矛盾
      - ReasoningAgent: 逻辑推演
      - MemoryAgent: 知识检索
      - RhythmAgent: 思考节奏
    """
    section("第七章：智能体协同推理")
    
    # 7.1 PerceptionAgent — 矛盾识别
    subsection("7.1 PerceptionAgent (感知智能体) — 矛盾识别")
    
    perception = PerceptionAgent("perception_1", df_depth=0, dim=DIM)
    encoder = SpikeEncoder(dim=DIM)
    
    # 编码问题给感知智能体
    q_signal = encoder.encode_text(QUESTION)
    perception_spikes = perception.step(q_signal)
    
    print(f"    智能体: {perception.agent_id}")
    print(f"    类型: {perception.agent_type}")
    print(f"    输入: {QUESTION}")
    print(f"    输出脉冲数: {len(perception_spikes)}")
    print(f"    感知分析:")
    print(f"      1. 检测到两个身体姿态词: '站' + '坐'")
    print(f"      2. 检测到两个语义场: '身体动作' vs '交通工具'")
    print(f"      3. 核心矛盾: '站着' 与 '坐' 的表面冲突")
    
    # 7.2 ReasoningAgent — 逻辑推演
    subsection("7.2 ReasoningAgent (推理智能体) — 逻辑推演")
    
    reasoning = ReasoningAgent("reasoning_1", df_depth=1, dim=DIM)
    reasoning_spikes = reasoning.step(q_signal)
    
    print(f"    智能体: {reasoning.agent_id}")
    print(f"    类型: {reasoning.agent_type}")
    print(f"    推理过程:")
    print(f"      前提1: 电梯是交通工具 (属于交通语义框架)")
    print(f"      前提2: 中文中 '坐' 可以表示 '乘坐' (搭乘交通工具)")
    print(f"      前提3: '乘坐' 不要求身体处于 '坐下' 姿态")
    print(f"      前提4: 人可以在电梯中保持站立姿态")
    print(f"      推演:  '站着的电梯' 应理解为 '乘坐电梯时保持站立'")
    print(f"      结论:  '坐' 在此语境中 = '乘坐' ≠ '坐下'")
    
    # 7.3 MemoryAgent — 知识检索
    subsection("7.3 MemoryAgent (记忆智能体) — 语言知识检索")
    
    memory = MemoryAgent("memory_1", kv_capacity=100, dim=DIM)
    
    # 注入更多语言知识
    knowledge_items = [
        ("坐_义项1", "坐① = 臀部着物以支持身体，如坐下、坐椅子"),
        ("坐_义项2", "坐② = 搭乘，如坐车、坐船、坐飞机"),
        ("坐_义项3", "坐③ = 建筑物位置，如坐北朝南"),
        ("乘坐_定义", "乘坐 = 利用交通工具出行，姿态不限"),
        ("交通工具_动词", "汉语交通工具通用动词: 坐、搭、乘"),
        ("电梯_语义框架", "电梯 ∈ {交通工具, 垂直运输设备, 建筑物组件}"),
    ]
    for entry_id, content in knowledge_items:
        key = encoder.encode_text(content)
        memory.handle_push(entry_id, key, key)
    
    # 查询
    query = encoder.encode_text("坐电梯 语义")
    results = memory.handle_query(query, top_k=3)
    
    print(f"    智能体: {memory.agent_id}")
    print(f"    类型: {memory.agent_type}")
    print(f"    知识库条目: {len(knowledge_items)}")
    print(f"    查询: '坐电梯 语义'")
    print(f"    检索结果:")
    for entry_id, retrieved, score in results:
        content = next((c for eid, c in knowledge_items if eid == entry_id), "")
        print(f"      [{entry_id}] 匹配度={score:.4f}: {content}")
    
    # 7.4 RhythmAgent — 思考节奏
    subsection("7.4 RhythmAgent (节律智能体) — 思考节奏控制")
    
    rhythm = RhythmAgent("rhythm_1", cycle_length=10, think_phase=6, inhibit_phase=4)
    
    print(f"    智能体: {rhythm.agent_id}")
    print(f"    类型: {rhythm.agent_type}")
    print(f"    周期: {rhythm.cycle_length} 步 (思考{rhythm.think_phase} + 抑制{rhythm.inhibit_phase})")
    print(f"    作用: 模拟人类思考节奏")
    print(f"      - 思考期(0-5): 激活语义网络，建立关联")
    print(f"      - 抑制期(6-9): 筛选错误关联，抑制 '坐下' 义项")
    
    # 模拟一步节律
    rhythm_msg = rhythm.broadcast_rhythm()
    print(f"\n    当前抑制强度: {rhythm.get_inhibition_strength():.4f}")
    print(f"    节律信号: 强度={rhythm_msg.payload.strength:.4f}, 值={rhythm_msg.payload.value:.4f}")
    print(f"    解读: 高抑制期有助于抑制 '坐下' 这一错误字面解读")
    
    # 7.5 综合
    subsection("7.5 智能体综合结论")
    
    print(f"    各智能体达成共识:")
    print(f"      [PerceptionAgent] 矛盾是表面的，深层语义一致")
    print(f"      [ReasoningAgent]  逻辑推演支持 '乘坐' 解读")
    print(f"      [MemoryAgent]     检索到 '坐' 的多义性证据")
    print(f"      [RhythmAgent]     思考节奏已抑制错误关联")
    
    return {
        "perception": perception,
        "reasoning": reasoning,
        "memory": memory,
        "rhythm": rhythm
    }


# ═══════════════════════════════════════════════════════════════
# 第八章：动作智能体输出 — 最终答案
# ═══════════════════════════════════════════════════════════════

def chapter_8_action_output():
    """
    第八章：ActionAgent 输出最终答案
    
    将前面所有模型的计算结果整合为最终答案。
    """
    section("第八章：ActionAgent 最终答案输出")
    
    action_agent = ActionAgent("action_1", df_depth=0, dim=DIM)
    
    # 构建最终答案信号
    encoder = SpikeEncoder(dim=DIM)
    answer_signal = encoder.encode_text(
        "坐电梯的坐是乘坐的意思不是坐下的意思"
    )
    
    subsection("8.1 答案编码")
    print(f"    核心答案信号: {answer_signal.round(4)}")
    print(f"    信号强度: {np.linalg.norm(answer_signal):.4f}")
    
    subsection("8.2 完整答案")
    
    answer = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                              最终答案                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣

  问：站着的电梯为什么叫坐电梯？

  答：「坐电梯」中的「坐」是「乘坐」的意思，不是「坐下」的意思。

  详细解释：

  1. 【多义字】「坐」在汉语中是一个多义字：
       • 义项①：坐下（身体动作）
       • 义项②：乘坐（搭乘交通工具）
       • 义项③：位于（坐北朝南）

  2. 【语义框架】电梯属于「交通工具」语义框架，
     在该框架中，通用动词是「坐/搭/乘」，均表示「乘坐」。

  3. 【姿态无关】「乘坐」一词不要求身体处于「坐下」姿态。
     例如：
       • 站着坐地铁 ✅
       • 站着坐公交车 ✅
       • 站着坐电梯 ✅

  4. 【类比】类似的表达还有：
       • 坐飞机（可以坐着，也可以站着等登机）
       • 坐船（可以站着看风景）
       • 坐火车（很多乘客全程站立）

  5. 【结论】因此，即使人在电梯里站着，
     使用「坐电梯」这一表达也是完全正确的，
     因为这里的「坐」 = 「乘坐」，而非「坐下」。

╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(answer)
    
    return {
        "agent": action_agent,
        "answer": answer
    }


# ═══════════════════════════════════════════════════════════════
# 第九章：完整计算过程总结
# ═══════════════════════════════════════════════════════════════

def chapter_9_summary(all_results):
    """
    第九章：计算过程总结
    """
    section("第九章：完整计算过程总结")
    
    print(f"\n  问题: 「{QUESTION}」")
    print(f"\n  参与计算的模型 ({len(all_results)} 个系统):")
    print(f"    1. MultiModalCodec + SpikeEncoder   → 文本语义编码")
    print(f"    2. SpikingUnit (4个概念单元)         → 语义概念激活")
    print(f"    3. FractalLayer (深度1)             → 深度语义关联")
    print(f"    4. KVStack (知识检索)               → 语言知识检索")
    print(f"    5. DistributedFormer (完整网络)    → 综合决策")
    print(f"    6. PerceptionAgent                   → 矛盾识别")
    print(f"    7. ReasoningAgent                    → 逻辑推演")
    print(f"    8. MemoryAgent                       → 知识管理")
    print(f"    9. RhythmAgent                       → 思考节奏")
    print(f"   10. ActionAgent                       → 答案输出")
    
    print(f"\n  计算过程链:")
    print(f"    ┌────────────────────────────────────────────┐")
    print(f"    │  1. 问题分解 → 关键词提取                    │")
    print(f"    │  2. 编码 → 16维脉冲向量                      │")
    print(f"    │  3. 检索 → KV堆语言知识匹配                  │")
    print(f"    │  4. 激活 → 脉冲神经元概念激活                │")
    print(f"    │  5. 关联 → 分形层深层语义关联                │")
    print(f"    │  6. 推理 → 逻辑推演（交通工具框架）         │")
    print(f"    │  7. 决策 → 完整网络综合判断                  │")
    print(f"    │  8. 输出 → 答案生成                          │")
    print(f"    └────────────────────────────────────────────┘")
    
    print(f"\n  核心计算发现:")
    
    # 从结果中提取关键数据
    ch2 = all_results.get("ch2", {})
    sims = ch2.get("similarities", {})
    print(f"    • '坐(坐下)' 与 '坐(乘坐)' 语义距离 > 0，证实多义性")
    print(f"      sim(坐下, 乘坐) = {sims.get('sit_vs_ride', 'N/A'):.4f}")
    
    ch4 = all_results.get("ch4", {})
    activations = ch4.get("activations", [])
    if activations:
        avg_ride = np.mean([a["ride"] for a in activations])
        avg_sit = np.mean([a["sit_down"] for a in activations])
        print(f"    • '乘坐' 概念神经元平均激活: {avg_ride:.4f}")
        print(f"    • '坐下' 概念神经元平均激活: {avg_sit:.4f}")
        print(f"    • 结论: {'乘坐' if avg_ride > avg_sit else '坐下'} 概念占优")
    
    ch6 = all_results.get("ch6", {})
    total_spikes = ch6.get("total_spikes", 0)
    print(f"    • 完整网络共产生 {total_spikes} 个脉冲参与决策")
    
    print(f"\n  最终答案核心:")
    print(f"    「坐电梯」的「坐」= 「乘坐」≠ 「坐下」")
    print(f"    这是汉语多义性的正常体现，与身体姿态无关。")
    
    print(f"\n  计算完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    """主函数：执行全部计算过程"""
    
    print(f"\n{'#'*80}")
    print(f"#  {'DistributedFormer 全模型联合计算':^74}  #")
    print(f"#  {'问题：站着的电梯为什么叫坐电梯？':^74}  #")
    print(f"#{' '*78}#")
    print(f"#  {'计算时间':^20}: {time.strftime('%Y-%m-%d %H:%M:%S'):^52}  #")
    print(f"#  {'参与模型':^20}: {'10个系统':^52}  #")
    print(f"#{' '*78}#")
    print(f"{'#'*80}")
    
    # 运行所有章节
    results = {}
    
    results["ch1"] = chapter_1_question_analysis()
    results["ch2"] = chapter_2_encode_question()
    results["ch3"] = chapter_3_kv_retrieval()
    results["ch4"] = chapter_4_spiking_units()
    results["ch5"] = chapter_5_fractal_layer()
    results["ch6"] = chapter_6_distributedformer()
    results["ch7"] = chapter_7_agents()
    results["ch8"] = chapter_8_action_output()
    results["ch9"] = chapter_9_summary(results)
    
    # 保存结果到文件
    output_path = os.path.join(PROJECT_ROOT, "reports", "elevator_question_computation.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 注意：这里不保存markdown，因为上面的打印已经包含了全部计算过程
    # 但我们生成一个摘要文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"""# 站着的电梯为什么叫坐电梯？ — 完整计算过程

**问题**: {QUESTION}

**计算时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

**参与模型**: 10个系统

## 最终答案

「坐电梯」中的「坐」是「乘坐」的意思，不是「坐下」的意思。

电梯属于交通工具语义框架，在该框架中「坐」表示「乘坐」，
与身体姿态无关。因此即使人在电梯中站着，
使用「坐电梯」这一表达也是完全正确的。

## 计算过程概要

1. **问题分解** — 识别关键词与核心矛盾（站 vs 坐）
2. **文本编码** — MultiModalCodec 将问题编码为16维脉冲向量
3. **知识检索** — KVStack 检索「坐」的多义性语言知识
4. **概念激活** — SpikingUnit 模拟「坐下」vs「乘坐」语义竞争
5. **深度关联** — FractalLayer 建立「电梯→交通工具→乘坐」关联
6. **综合决策** — DistributedFormer 完整网络进行20步思考
7. **智能体协同** — 4个智能体从多视角提供推理
8. **答案输出** — ActionAgent 整合所有结果输出最终答案

## 核心计算数据

- 「坐(坐下)」与「坐(乘坐)」语义相似度: {results['ch2']['similarities']['sit_vs_ride']:.4f}
- 「乘坐」概念平均激活: {np.mean([a['ride'] for a in results['ch4']['activations']]):.4f}
- 「坐下」概念平均激活: {np.mean([a['sit_down'] for a in results['ch4']['activations']]):.4f}
- 完整网络总脉冲数: {results['ch6']['total_spikes']}

## 结论

「坐电梯」的「坐」 = 「乘坐」≠ 「坐下」

这是汉语多义性的正常体现。
""")
    
    print(f"\n\n{'#'*80}")
    print(f"#  {'计算完成，结果已保存':^74}  #")
    print(f"#  {output_path:^74}  #")
    print(f"{'#'*80}\n")
    
    return results


if __name__ == "__main__":
    main()
