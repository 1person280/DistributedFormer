"""
训练数据生成器 - 为 DistributedFormer 生成带标签的合成训练数据

基于股票监控场景，生成4类脉冲模式：
  0: normal   - 正常波动
  1: uptrend  - 上涨趋势
  2: downtrend- 下跌趋势
  3: anomaly  - 异常波动
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributedformer.codec.spike_codec import MultiModalCodec


@dataclass
class TrainingSample:
    """单个训练样本"""
    sample_id: str
    category: int           # 0=normal, 1=uptrend, 2=downtrend, 3=anomaly
    category_name: str
    input_signal: np.ndarray     # 16维输入脉冲
    target_pattern: np.ndarray   # 16维期望输出模式
    metadata: Dict


class StockTrainingDataset:
    """
    合成股票训练数据集
    
    生成逻辑:
    - 正常: 涨跌幅 [-2%, +2%], 成交量正常, 新闻中性
    - 上涨: 涨跌幅 [+3%, +10%], 成交量放大, 新闻积极
    - 下跌: 涨跌幅 [-10%, -3%], 成交量放大, 新闻消极
    - 异常: 涨跌幅绝对值 >8%, 成交量异常, 新闻含警报词
    
    监督信号（目标输出模式）:
    - 正常  → 输出维度0高激活 (低强度)
    - 上涨  → 输出维度1高激活
    - 下跌  → 输出维度2高激活
    - 异常  → 输出维度3高激活 (高强度)
    """
    
    CATEGORIES = ["normal", "uptrend", "downtrend", "anomaly"]
    
    # 新闻词库
    NEWS_POSITIVE = [
        "strong earnings", "beats estimates", "revenue growth",
        "new product launch", "partnership announced", "buy rating upgrade",
        "AI breakthrough", "market share gains", "dividend increase"
    ]
    NEWS_NEGATIVE = [
        "misses estimates", "revenue decline", "layoffs announced",
        "regulatory scrutiny", "sell rating", "supply chain issues",
        "lawsuit filed", "guidance cut", "debt concerns"
    ]
    NEWS_NEUTRAL = [
        "trading volume normal", "market close update", "analyst report",
        "quarterly review", "board meeting", "routine filing",
        "market opens flat", "sector performance mixed"
    ]
    NEWS_ALARM = [
        "trading halted", "SEC investigation", "CEO resigns",
        "accounting irregularities", "merger cancelled", "bankruptcy filing",
        "massive sell-off", "circuit breaker triggered"
    ]
    
    def __init__(self, dim: int = 16, seed: int = 42):
        self.dim = dim
        self.codec = MultiModalCodec(dim=dim)
        random.seed(seed)
        np.random.seed(seed)
        
    def _generate_normal(self, ticker: str = "STOCK") -> Tuple[Dict, str]:
        """生成正常波动数据"""
        price = random.uniform(100, 500)
        change_pct = random.gauss(0, 1.5)  # 均值0，标准差1.5
        volume = random.uniform(20_000_000, 60_000_000)
        news = random.choice(self.NEWS_NEUTRAL)
        if random.random() < 0.3:
            news = random.choice(self.NEWS_POSITIVE if random.random() < 0.5 else self.NEWS_NEGATIVE)
        
        return {
            "price": price,
            "change_pct": change_pct,
            "volume": volume,
            "ticker": ticker
        }, news
    
    def _generate_uptrend(self, ticker: str = "STOCK") -> Tuple[Dict, str]:
        """生成上涨趋势数据"""
        price = random.uniform(100, 500)
        change_pct = random.uniform(3.0, 10.0)
        volume = random.uniform(50_000_000, 120_000_000)
        news = random.choice(self.NEWS_POSITIVE)
        if random.random() < 0.3:
            news += ", " + random.choice(self.NEWS_POSITIVE)
        
        return {
            "price": price,
            "change_pct": change_pct,
            "volume": volume,
            "ticker": ticker
        }, news
    
    def _generate_downtrend(self, ticker: str = "STOCK") -> Tuple[Dict, str]:
        """生成下跌趋势数据"""
        price = random.uniform(100, 500)
        change_pct = random.uniform(-10.0, -3.0)
        volume = random.uniform(50_000_000, 120_000_000)
        news = random.choice(self.NEWS_NEGATIVE)
        if random.random() < 0.3:
            news += ", " + random.choice(self.NEWS_NEGATIVE)
        
        return {
            "price": price,
            "change_pct": change_pct,
            "volume": volume,
            "ticker": ticker
        }, news
    
    def _generate_anomaly(self, ticker: str = "STOCK") -> Tuple[Dict, str]:
        """生成异常波动数据"""
        price = random.uniform(100, 500)
        change_pct = random.uniform(-20.0, -8.0) if random.random() < 0.5 else random.uniform(8.0, 20.0)
        volume = random.uniform(100_000_000, 500_000_000)
        news = random.choice(self.NEWS_ALARM)
        if random.random() < 0.5:
            news += ", " + random.choice(self.NEWS_NEGATIVE)
        
        return {
            "price": price,
            "change_pct": change_pct,
            "volume": volume,
            "ticker": ticker
        }, news
    
    def _make_target_pattern(self, category: int) -> np.ndarray:
        """
        生成监督目标输出模式
        
        维度映射:
        - dim 0: normal (低强度, ~0.3)
        - dim 1: uptrend (中等, ~0.6)
        - dim 2: downtrend (中等, ~0.6)
        - dim 3: anomaly (高强度, ~0.9)
        - dim 4-15: 噪声基底 (~0.05)
        """
        pattern = np.ones(self.dim) * 0.05  # 基底噪声
        
        if category == 0:  # normal
            pattern[0] = 0.35 + random.gauss(0, 0.05)
            pattern[4] = 0.2 + random.gauss(0, 0.03)
        elif category == 1:  # uptrend
            pattern[1] = 0.65 + random.gauss(0, 0.08)
            pattern[5] = 0.3 + random.gauss(0, 0.05)
        elif category == 2:  # downtrend
            pattern[2] = 0.65 + random.gauss(0, 0.08)
            pattern[6] = 0.3 + random.gauss(0, 0.05)
        elif category == 3:  # anomaly
            pattern[3] = 0.9 + random.gauss(0, 0.05)
            pattern[7] = 0.5 + random.gauss(0, 0.05)
            pattern[0] = 0.15  # 异常时也有微弱normal信号
        
        return np.clip(pattern, 0.0, 1.0)
    
    def generate_sample(self, category: int, ticker: str = "STOCK") -> TrainingSample:
        """生成单个训练样本"""
        generators = [
            self._generate_normal,
            self._generate_uptrend,
            self._generate_downtrend,
            self._generate_anomaly
        ]
        
        data, news = generators[category](ticker)
        
        # 编码为脉冲信号
        input_signal = self.codec.encode_stock_data(
            price=data["price"],
            change_pct=data["change_pct"],
            volume=data["volume"],
            news_text=news
        )
        
        # 目标输出模式
        target_pattern = self._make_target_pattern(category)
        
        return TrainingSample(
            sample_id=f"{self.CATEGORIES[category]}_{random.randint(10000, 99999)}",
            category=category,
            category_name=self.CATEGORIES[category],
            input_signal=input_signal,
            target_pattern=target_pattern,
            metadata={
                "price": data["price"],
                "change_pct": data["change_pct"],
                "volume": data["volume"],
                "news": news
            }
        )
    
    def generate_dataset(self, samples_per_class: int = 100,
                         tickers: List[str] = None,
                         train_ratio: float = 0.8) -> Tuple[List[TrainingSample], List[TrainingSample]]:
        """
        生成完整数据集并划分训练/验证集
        
        Returns:
            (train_samples, val_samples)
        """
        tickers = tickers or ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
        all_samples: List[TrainingSample] = []
        
        for cat in range(4):
            for i in range(samples_per_class):
                ticker = tickers[i % len(tickers)]
                sample = self.generate_sample(cat, ticker)
                all_samples.append(sample)
        
        # 打乱
        random.shuffle(all_samples)
        
        # 划分
        split_idx = int(len(all_samples) * train_ratio)
        train = all_samples[:split_idx]
        val = all_samples[split_idx:]
        
        return train, val
    
    def get_class_distribution(self, samples: List[TrainingSample]) -> Dict:
        """统计类别分布"""
        counts = {name: 0 for name in self.CATEGORIES}
        for s in samples:
            counts[s.category_name] += 1
        return counts


# ═══════════════════════════════════════════════════════════════
# 自测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("训练数据生成器测试")
    print("=" * 60)
    
    dataset = StockTrainingDataset(dim=16, seed=42)
    
    # 生成每类1个样本看看
    print("\n[各类样本示例]")
    for cat in range(4):
        sample = dataset.generate_sample(cat)
        print(f"\n  类别 {cat} ({sample.category_name}):")
        print(f"    ID: {sample.sample_id}")
        print(f"    价格: ${sample.metadata['price']:.2f}")
        print(f"    涨跌幅: {sample.metadata['change_pct']:+.2f}%")
        print(f"    成交量: {sample.metadata['volume']:,.0f}")
        print(f"    新闻: {sample.metadata['news'][:60]}...")
        print(f"    输入信号非零维: {np.count_nonzero(sample.input_signal)}")
        print(f"    目标模式: [{', '.join(f'{v:.2f}' for v in sample.target_pattern[:4])}...]")
    
    # 生成完整数据集
    print("\n[生成完整数据集]")
    train, val = dataset.generate_dataset(samples_per_class=50)
    print(f"  训练集: {len(train)} 样本")
    print(f"  验证集: {len(val)} 样本")
    print(f"  训练集分布: {dataset.get_class_distribution(train)}")
    print(f"  验证集分布: {dataset.get_class_distribution(val)}")
    
    print("\n" + "=" * 60)
    print("数据生成器测试通过!")
    print("=" * 60)
