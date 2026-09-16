"""pytest 根配置: 统一 Python 字节码缓存到根目录 ./cache/pycache

src/ 下 11 个子包各自生成 __pycache__ 过于分散 (v0.8.7 精简),
在测试收集前设置 sys.pycache_prefix, 之后导入的 src/tests 模块
的字节码统一写入 ./cache/pycache, 不再散落各子目录。
直接运行脚本由 src/__init__.py 导入期同样设置;
自定义位置可用环境变量 PYTHONPYCACHEPREFIX。
"""

import os
import sys

_PYNCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cache", "pycache")
os.makedirs(_PYNCACHE, exist_ok=True)
sys.pycache_prefix = _PYNCACHE
