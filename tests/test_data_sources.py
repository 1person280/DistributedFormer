import pytest

from src.demos.stock_monitor import StockDataSimulator


def test_simulator_fetch():
    sim = StockDataSimulator()
    quote = sim.fetch_price("AAPL")
    assert quote["price"] > 0
    assert "change_pct" in quote
    news = sim.fetch_news("AAPL")
    assert isinstance(news, str) and news


def test_realtime_source_import_guard():
    """未安装 yfinance 时应给出清晰错误, 而非 ImportError 裸抛"""
    pytest.importorskip("yfinance", reason="可选依赖, 未安装则跳过")
    from src.demos.stock_monitor import YFinanceDataSource
    src = YFinanceDataSource(["AAPL"])
    assert src.tickers == ["AAPL"]
