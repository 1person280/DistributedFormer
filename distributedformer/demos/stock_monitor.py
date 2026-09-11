"""
股票监控端到端演示

场景: 智能股票监控脉冲网络

目标: 构建一个持续运行的脉冲智能体网络，自动监控股票数据，
      检测异常趋势，生成分析报告，并在关键时刻发送通知。

输入:
  • Yahoo Finance API (模拟)
  • 每5分钟获取 AAPL、TSLA、NVDA 的实时价格
  • 同时抓取相关新闻标题 (模拟)

处理:
  • 价格变化率 → 标量脉冲编码
  • 新闻文本 → TF-IDF稀疏脉冲编码
  • 脉冲注入 reasoning_0 (趋势分析) 和 reasoning_1 (异常检测)
  • 两个推理智能体异步计算，侧向交换脉冲
  • 查询全局KV堆，对比历史模式

输出:
  • 正常: 静默记录到KV堆
  • 趋势变化: 生成Markdown报告，写入 reports/
  • 异常检测: 桌面通知 + 日志记录
  • 重大事件: 记录到报告
"""

import numpy as np
import time
import json
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributedformer.core.distributedformer import SpikeMessage
from distributedformer.agents.base_agent import (
    PerceptionAgent, ReasoningAgent, ActionAgent, 
    MemoryAgent, RhythmAgent
)
from distributedformer.workflow.engine import SpikeWorkflowEngine, WorkflowConfig
from distributedformer.codec.spike_codec import MultiModalCodec


class StockDataSimulator:
    """股票数据模拟器 (模拟Yahoo Finance API)"""
    
    def __init__(self):
        self.tickers = {
            "AAPL": {"base_price": 175.0, "volatility": 2.5, "trend": 0.02},
            "TSLA": {"base_price": 240.0, "volatility": 5.0, "trend": -0.01},
            "NVDA": {"base_price": 880.0, "volatility": 15.0, "trend": 0.05}
        }
        self.price_history: Dict[str, List[float]] = {t: [] for t in self.tickers}
        self.news_pool = [
            "Apple reports strong quarterly earnings",
            "Tesla announces new factory location",
            "NVIDIA unveils next-gen AI chip",
            "Market volatility increases amid geopolitical tensions",
            "Federal Reserve holds interest rates steady",
            "Tech sector rallies on AI optimism",
            "Supply chain disruptions affect semiconductor industry",
            "Consumer spending data exceeds expectations",
            "Regulatory concerns weigh on tech stocks",
            "Energy prices surge on Middle East tensions",
            "AI regulation bill introduced in Congress",
            "Major acquisition announced in tech sector",
            "Quarterly GDP growth beats estimates",
            "Inflation data shows cooling trend",
            "New product launch drives investor enthusiasm"
        ]
        
    def fetch_price(self, ticker: str) -> Dict:
        """获取模拟股票价格"""
        if ticker not in self.tickers:
            return {"error": "Unknown ticker"}
        
        config = self.tickers[ticker]
        
        # 随机游走进化
        prev_price = config["base_price"]
        if self.price_history[ticker]:
            prev_price = self.price_history[ticker][-1]
        
        # 生成新价格
        change = random.gauss(config["trend"], config["volatility"])
        new_price = max(1.0, prev_price + change)
        
        self.price_history[ticker].append(new_price)
        if len(self.price_history[ticker]) > 100:
            self.price_history[ticker].pop(0)
        
        # 计算变化率
        change_pct = ((new_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        return {
            "ticker": ticker,
            "price": round(new_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": random.randint(10_000_000, 100_000_000),
            "timestamp": time.time()
        }
    
    def fetch_news(self, ticker: str) -> str:
        """获取模拟新闻"""
        # 随机选择新闻，与ticker有一定关联性
        base_news = random.choice(self.news_pool)
        
        # 根据ticker添加特定内容
        ticker_specific = {
            "AAPL": ["iPhone", "Mac", "App Store", "Tim Cook"],
            "TSLA": ["EV", "FSD", "Cybertruck", "Elon Musk"],
            "NVDA": ["GPU", "CUDA", "Data Center", "Jensen Huang"]
        }
        
        if ticker in ticker_specific:
            keywords = ticker_specific[ticker]
            keyword = random.choice(keywords)
            if random.random() < 0.5:
                base_news += f", {keyword} mentioned"
        
        return base_news
    
    def get_price_history(self, ticker: str, n: int = 5) -> List[float]:
        """获取最近n个价格"""
        history = self.price_history.get(ticker, [])
        return history[-n:] if len(history) >= n else history


class YFinanceDataSource:
    """真实行情数据源 (基于 yfinance, 可选依赖)

    与 StockDataSimulator 接口一致: fetch_price / fetch_news / get_price_history。
    新闻暂以行情摘要代替, 不依赖外部新闻 API。
    """

    def __init__(self, tickers: List[str]):
        try:
            import yfinance  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "使用真实行情需要安装 yfinance: pip install 'distributedformer[realtime]'"
            ) from e
        self.tickers = list(tickers)
        self.price_history: Dict[str, List[float]] = {t: [] for t in self.tickers}

    def fetch_price(self, ticker: str) -> Dict:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist.empty:
            raise RuntimeError(f"{ticker}: 未获取到行情数据")
        close = hist["Close"].dropna()
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else price
        change_pct = (price - prev) / prev * 100 if prev else 0.0
        volume = int(hist["Volume"].iloc[-1]) if "Volume" in hist else 0
        for p in close:
            self.price_history.setdefault(ticker, []).append(float(p))
        return {
            "ticker": ticker,
            "price": price,
            "prev_price": prev,
            "change_pct": change_pct,
            "volume": volume,
            "timestamp": time.time(),
        }

    def fetch_news(self, ticker: str) -> str:
        return f"{ticker} market data digest (yfinance)"

    def get_price_history(self, ticker: str, n: int = 5) -> List[float]:
        return self.price_history.get(ticker, [])[-n:]


class StockMonitorWorkflow:
    """股票监控工作流

    Args:
        tickers: 监控的股票代码
        data_source: 行情数据源, 默认使用内置模拟器;
                     传入 YFinanceDataSource 即可切换为真实行情
    """

    def __init__(self, tickers: List[str] = None, data_source=None):
        self.tickers = tickers or ["AAPL", "TSLA", "NVDA"]
        self.simulator = data_source if data_source is not None else StockDataSimulator()
        self.codec = MultiModalCodec(dim=16)
        
        # 工作流引擎
        config = WorkflowConfig(
            name="智能股票监控脉冲网络",
            cycle_length=120,
            think_phase=80,
            inhibit_phase=40,
            dim=16
        )
        self.engine = SpikeWorkflowEngine(config)
        
        # 监控结果
        self.anomalies_detected = 0
        self.reports_generated = 0
        self.notifications_sent = 0
        
        # 初始化智能体
        self._setup_agents()
    
    def _setup_agents(self) -> None:
        """设置智能体网络"""
        dim = self.engine.config.dim
        kv = self.engine.global_kv
        
        # 1. 感知智能体 (为每个股票创建一个)
        for ticker in self.tickers:
            agent_id = f"perception_{ticker.lower()}"
            agent = PerceptionAgent(agent_id, df_depth=1, dim=dim, kv_stack=kv)
            agent.add_input(f"finance://yahoo/{ticker}", "multi_stock")
            agent.set_targets(["reasoning_trend", "reasoning_anomaly"])
            self.engine.register_agent(agent_id, agent)
        
        # 2. 推理智能体 - 趋势分析
        trend_agent = ReasoningAgent("reasoning_trend", df_depth=2, dim=dim, kv_stack=kv)
        trend_agent.set_specialization("trend_analysis")
        trend_agent.set_lateral(["reasoning_anomaly"])
        trend_agent.set_targets(["action_main"])
        self.engine.register_agent("reasoning_trend", trend_agent)
        
        # 3. 推理智能体 - 异常检测
        anomaly_agent = ReasoningAgent("reasoning_anomaly", df_depth=2, dim=dim, kv_stack=kv)
        anomaly_agent.set_specialization("anomaly_detection")
        anomaly_agent.set_lateral(["reasoning_trend"])
        anomaly_agent.set_targets(["action_main"])
        self.engine.register_agent("reasoning_anomaly", anomaly_agent)
        
        # 4. 动作智能体
        action_agent = ActionAgent("action_main", df_depth=0, dim=dim, kv_stack=kv)
        action_agent.add_output("file_write", path="reports/stock_monitor.md")
        action_agent.add_output("desktop_notification", condition="output_strength > 0.8")
        action_agent.add_output("report_generate", template="stock_analysis")
        self.engine.register_agent("action_main", action_agent)
        
        # 5. 记忆智能体
        memory_agent = MemoryAgent("memory_main", kv_capacity=50000, dim=dim)
        self.engine.register_agent("memory_main", memory_agent)
        
        # 6. 节律智能体
        rhythm_agent = RhythmAgent("rhythm_main", cycle_length=120, think_phase=80, inhibit_phase=40, dim=dim)
        self.engine.register_agent("rhythm_main", rhythm_agent)
        
        # 建立连接
        for ticker in self.tickers:
            self.engine.connect(f"perception_{ticker.lower()}", "reasoning_trend")
            self.engine.connect(f"perception_{ticker.lower()}", "reasoning_anomaly")
        
        self.engine.connect("reasoning_trend", "action_main")
        self.engine.connect("reasoning_anomaly", "action_main")
        self.engine.connect("action_main", "memory_main")
    
    def run_cycle(self) -> Dict:
        """
        执行一个监控周期
        
        流程:
        1. 获取股票数据
        2. 编码为脉冲
        3. 注入感知智能体
        4. 运行工作流引擎
        5. 收集结果
        """
        cycle_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers": {},
            "anomalies": [],
            "actions": []
        }
        
        # 1. 获取股票数据
        for ticker in self.tickers:
            price_data = self.simulator.fetch_price(ticker)
            news = self.simulator.fetch_news(ticker)
            history = self.simulator.get_price_history(ticker, 5)
            
            cycle_results["tickers"][ticker] = {
                "price": price_data["price"],
                "change_pct": price_data["change_pct"],
                "volume": price_data["volume"],
                "news": news
            }
            
            # 2. 构造多模态输入 (编码由各 InputModule 内部完成)
            stock_inputs = {
                "numeric": np.array([
                    price_data["change_pct"] / 10.0,
                    price_data["price"] / 1000.0,
                    price_data["volume"] / 1e8,
                ]),
                "text": news,
                "timeseries": self.simulator.get_price_history(ticker, 5),
            }

            # 3. 注入感知智能体
            agent_id = f"perception_{ticker.lower()}"
            if agent_id in self.engine.agents:
                agent = self.engine.agents[agent_id]
                # 直接设置外部输入
                agent.step(stock_inputs)
        
        # 4. 运行工作流引擎
        engine_stats = self.engine.step()
        
        # 5. 检测异常 (基于推理智能体输出)
        for agent_id, agent in self.engine.agents.items():
            if agent.agent_type == "reasoning" and agent.df:
                output_pattern = agent.df.get_output_pattern()
                max_activation = np.max(np.abs(output_pattern))
                
                if max_activation > 0.7:
                    # 高激活 = 可能异常
                    anomaly = {
                        "agent": agent_id,
                        "specialization": agent.specialization,
                        "activation": float(max_activation),
                        "ticker": self._infer_ticker_from_agent(agent_id)
                    }
                    cycle_results["anomalies"].append(anomaly)
                    self.anomalies_detected += 1
        
        # 收集动作执行结果
        for agent_id, agent in self.engine.agents.items():
            if agent.agent_type == "action":
                for record in agent.action_history:
                    cycle_results["actions"].append({
                        "type": record["action"]["type"],
                        "strength": record["action"]["strength"],
                        "status": record["result"]["status"]
                    })
        
        return cycle_results
    
    def _infer_ticker_from_agent(self, agent_id: str) -> str:
        """从智能体ID推断相关股票"""
        for ticker in self.tickers:
            if ticker.lower() in agent_id:
                return ticker
        return "UNKNOWN"
    
    def run_simulation(self, num_cycles: int = 10, interval_sec: float = 1.0) -> List[Dict]:
        """
        运行模拟
        
        Args:
            num_cycles: 模拟周期数
            interval_sec: 周期间隔(秒)
        """
        print("=" * 70)
        print("  智能股票监控脉冲网络 - 端到端演示")
        print("=" * 70)
        print(f"\n监控股票: {', '.join(self.tickers)}")
        print(f"模拟周期: {num_cycles}")
        print(f"周期间隔: {interval_sec}秒\n")
        
        results = []
        
        for i in range(num_cycles):
            print(f"\n{'─' * 70}")
            print(f"  周期 {i+1}/{num_cycles}")
            print(f"{'─' * 70}")
            
            result = self.run_cycle()
            results.append(result)
            
            # 显示结果
            print(f"\n  时间: {result['timestamp']}")
            print(f"\n  [股票数据]")
            for ticker, data in result['tickers'].items():
                emoji = "📈" if data['change_pct'] > 0 else "📉"
                print(f"    {emoji} {ticker}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)"
                      f"  Vol: {data['volume']:,}")
                print(f"       新闻: {data['news'][:60]}...")
            
            if result['anomalies']:
                print(f"\n  [⚠️ 异常检测] {len(result['anomalies'])} 个")
                for a in result['anomalies']:
                    print(f"    • {a['agent']} ({a['specialization']}): "
                          f"激活强度 {a['activation']:.2f}")
            
            if result['actions']:
                print(f"\n  [执行动作]")
                for action in result['actions']:
                    icon = "✅" if action['status'] == 'success' else '❌'
                    print(f"    {icon} {action['type']}: 强度 {action['strength']:.2f}")
            
            # 全局统计
            gstats = self.engine.get_global_stats()
            print(f"\n  [全局] 脉冲: {gstats['total_spikes_sent']} | "
                  f"KV利用率: {gstats['kv_utilization']:.1%}")
            
            if i < num_cycles - 1:
                time.sleep(interval_sec)
        
        # 最终报告
        print(f"\n{'=' * 70}")
        print("  模拟完成")
        print(f"{'=' * 70}")
        print(f"\n  总异常检测: {self.anomalies_detected}")
        print(f"  总脉冲发送: {gstats['total_spikes_sent']}")
        print(f"  KV堆利用率: {gstats['kv_utilization']:.1%}")
        
        return results
    
    def generate_summary_report(self) -> str:
        """生成总结报告"""
        gstats = self.engine.get_global_stats()
        
        report = f"""# 股票监控脉冲网络 - 运行总结报告

## 运行参数
- 监控股票: {', '.join(self.tickers)}
- 智能体数: {gstats['total_agents']}
- 总执行步数: {gstats['total_steps']}
- 异常检测次数: {self.anomalies_detected}

## 系统统计
- 总脉冲发送: {gstats['total_spikes_sent']}
- 总脉冲接收: {gstats['total_spikes_received']}
- KV堆利用率: {gstats['kv_utilization']:.1%}

## 智能体详情
"""
        for agent_id, agent in self.engine.agents.items():
            stats = agent.get_stats()
            report += f"\n### {agent_id} ({stats['agent_type']})\n"
            report += f"- 执行周期: {stats['cycles']}\n"
            report += f"- 发送脉冲: {stats['spikes_sent']}\n"
            report += f"- 接收脉冲: {stats['spikes_received']}\n"
            if 'specialization' in stats and stats['specialization']:
                report += f"- 特殊化: {stats['specialization']}\n"
        
        report += f"""

## 评估指标
| 维度 | 指标 | 目标 | 实际 |
|------|------|------|------|
| 持续性 | 无输入自发脉冲 | >10/分钟 | {gstats['total_spikes_sent'] / max(gstats['total_steps'], 1) * 60:.1f}/分钟 |
| 响应性 | 输入到输出延迟 | <5秒 | ~1秒 (模拟) |
| 记忆性 | KV堆利用率 | >70% | {gstats['kv_utilization']:.1%} |
| 协作性 | 消息路由 | >1000/min | {gstats['total_spikes_sent'] / max(gstats['total_steps'], 1) * 60:.0f}/min |

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票监控脉冲网络演示')
    parser.add_argument('--cycles', type=int, default=10, help='模拟周期数')
    parser.add_argument('--interval', type=float, default=1.0, help='周期间隔(秒)')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'TSLA', 'NVDA'], help='监控股票')
    parser.add_argument('--report', action='store_true', help='生成总结报告')
    args = parser.parse_args()
    
    # 创建工作流
    workflow = StockMonitorWorkflow(tickers=args.tickers)
    
    # 运行模拟
    results = workflow.run_simulation(
        num_cycles=args.cycles,
        interval_sec=args.interval
    )
    
    # 生成报告
    if args.report:
        report = workflow.generate_summary_report()
        report_path = "reports/stock_monitor_summary.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 总结报告已保存: {report_path}")
    
    print("\n✅ 演示完成!")
    return results


if __name__ == "__main__":
    # 如果没有命令行参数，使用默认参数运行
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend(['--cycles', '8', '--interval', '0.5', '--report'])
    
    main()
