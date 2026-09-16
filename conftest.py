"""pytest 根配置: 仓库内零 __pycache__ (v0.8.7)

Python 字节码缓存曾散落在 src/ 各子包与根目录; 曾用
sys.pycache_prefix 统一到 cache/pycache, 但该机制按源文件
绝对路径镜像目录树 (又长又臭) 且会把 stdlib 一起复制进来,
故弃用。现改为:

    sys.dont_write_bytecode = True
    (测试收集前设置, 之后导入的 src/tests 模块一律不落地 .pyc)

本文件自身的 .pyc 在设置前已被写入根目录 __pycache__,
由 atexit 在退出时清掉 —— 运行结束后仓库内不存在任何
__pycache__。如需保留字节码缓存, 在运行前设
PYTHONPYCACHEPREFIX 指向仓库外的目录即可。
"""

import atexit
import os
import shutil
import sys

sys.dont_write_bytecode = True

# 清掉本 conftest 自身刚落地的字节码 (写入早于上面的开关)
atexit.register(shutil.rmtree, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "__pycache__"),
    ignore_errors=True)
