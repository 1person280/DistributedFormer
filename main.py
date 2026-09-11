"""向后兼容入口: 推荐使用 `python -m distributedformer.cli` 或 `df` 命令"""

from distributedformer.cli import main

if __name__ == "__main__":
    main()
