"""TagSearchingPlugin: 代码引用检索思考插件 (v0.14.5)

给定一个"标签/记录" (函数名、类名、符号), 扫描宿主代码库并**准确**
给出引用情况: 定义点 + 全部引用点 (文件 + 行号 + 上下文片段), 并区分
"定义" 与 "调用/使用"。路由 topic = "tagsearch"。

    事件 data 支持:
        "foo"                                    → 检索符号 foo (默认扫 ./src)
        {"tag": "foo", "root": "...", "exts": [...]}  → 指定根目录/扩展名

检索是纯本地确定性文本分析 (与训练无关, 无网络 / 无幻觉):
    - 标识符**词边界**精确匹配 (前后都不是字母/数字/下划线), 杜绝子串误报
    - 定义行启发式 (def/fn/function/class/struct/impl/type/赋值/注解),
      其余命中按"引用"归类
    - 返回结构化报告 + 语义记忆留痕 (recall_search 可复查)

默认扫描仓库真实代码 ./src (也接受显式 root); 跳过隐藏目录与打包产物。
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .plugin import ExpertPlugin, PluginContext

# 覆盖常见宿主语言源文件
DEFAULT_EXTS = (".py", ".rs", ".java", ".js", ".ts", ".jsx", ".tsx",
                ".go", ".c", ".h", ".cpp", ".hpp")
_SKIP_DIRS = {".git", ".github", "__pycache__", "cache", "node_modules",
              "dist", "build", "docs", "plugin"}

# 各语言"定义行"启发式: 定义关键字紧跟标签 (函数/类/结构/接口/枚举/trait/impl/type)
_DEF_KW_RE = re.compile(
    r"(?:def|fn|func|function|class|struct|interface|enum|trait|impl|type)"
    r"\s+{tag}\b")
# 赋值式定义: 行首 {tag} = (排除 ==); 注解式: 行首 {tag}: (排除 :: 与 :=)
_ASSIGN_RE = re.compile(r"^\s*{tag}\s*=\s*[^=\s]")
_ANNOT_RE = re.compile(r"^\s*{tag}\s*:\s*[^=:]")

_IDENT_BOUNDARY = re.compile(r"(?<![\w]){tag}(?![\w])")


class TagSearchingPlugin(ExpertPlugin):
    """给定符号 → 定义点 + 引用点 (文件/行号/片段), 引用情况一目了然

    事件格式: {"topic": "tagsearch", "data": <tag str 或 {"tag": ..., ...}>}
    返回: {"tag", "definitions", "references", "reference_count",
           "definition_count", "scanned_files", "root"}。
    并广播 "tagsearch.result" (只广播摘要)。
    """

    BASE_MODEL = "tag.searching"
    CAPABILITY = ("代码引用检索: 给定函数/类/符号, 精确返回定义点与全部"
                  "引用点 (文件+行号+片段), 区分定义与调用; 纯本地确定性")

    def __init__(self, name: str = "tag-search", *,
                 route: Optional[str] = None, **kwargs):
        super().__init__(name, route=route or "tagsearch", **kwargs)
        self.search_count = 0
        # 项目根 (src/cutemamen/tag_searching.py → 上溯 3 级 = 仓库根)
        self.repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        self._tag_cache: Dict[str, Dict[str, Any]] = {}

    # ── 生命周期 ────────────────────────────────────────────
    def on_load(self, ctx: PluginContext) -> None:
        self.memory.set("base_model", self.BASE_MODEL)
        self.memory.set("default_root", "src")
        self.memory.set("exts", list(DEFAULT_EXTS))
        super().on_load(ctx)

    def on_think(self, event: Dict[str, Any],
                 ctx: PluginContext) -> Optional[Dict[str, Any]]:
        """检索一个符号的引用情况"""
        super().on_think(event, ctx)
        query = event.get("data")
        tag, root, exts = _parse_query(query)
        if not tag:
            return None
        report = self.search(tag, root=root, exts=exts)
        self.search_count += 1
        self.memory.remember(f"tag:{tag}", {
            "definitions": len(report["definitions"]),
            "references": len(report["references"]),
            "root": report["root"],
            "scanned_files": report["scanned_files"],
        })
        if ctx is not None:
            ctx.emit("tagsearch.result", {
                "tag": tag, "definitions": len(report["definitions"]),
                "references": len(report["references"]),
                "scanned_files": report["scanned_files"],
            })
        return report

    def on_unload(self) -> None:
        self.memory.consolidate()
        super().on_unload()

    # ── 检索入口 ────────────────────────────────────────────
    def search(self, tag: str, root: Optional[str] = None,
               exts: Optional[List[str]] = None) -> Dict[str, Any]:
        """检索符号 → 结构化引用报告 (按文件+行号排序)

        - 定义: def/fn/function/class/struct/interface/enum/trait/type/
          impl 前缀, 或行首赋值/注解式 ({tag} = / : ...)
        - 引用: 其余词边界命中
        默认 root = 仓库 ./src; exts 缺省用 DEFAULT_EXTS。
        """
        tag = str(tag).strip()
        if not tag or not _is_identifier(tag):
            raise ValueError(
                f"无效标签 {tag!r}: 需为合法标识符 (字母/数字/下划线)")
        search_root = root or os.path.join(self.repo_root, "src")
        exts = tuple(exts or DEFAULT_EXTS)

        pattern = _IDENT_BOUNDARY.pattern.format(tag=re.escape(tag))
        token_re = re.compile(pattern)
        def_kw_re = re.compile(
            _DEF_KW_RE.pattern.format(tag=re.escape(tag)))
        assign_re = re.compile(_ASSIGN_RE.pattern.format(tag=re.escape(tag)))
        annot_re = re.compile(_ANNOT_RE.pattern.format(tag=re.escape(tag)))

        definitions: List[Dict[str, Any]] = []
        references: List[Dict[str, Any]] = []
        scanned = 0

        for file_path, rel in _iter_source(search_root, exts):
            scanned += 1
            try:
                with open(file_path, "r", encoding="utf-8",
                          errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            for lineno, raw in enumerate(lines, start=1):
                line = raw.rstrip("\n")
                m = token_re.search(line)
                if not m:
                    continue
                # 定义 = 关键字定义 或 行首赋值/注解 (排除 == / :: / :=)
                is_def = bool(
                    def_kw_re.search(line)
                    or assign_re.match(line)
                    or annot_re.match(line))
                start_col = max(0, m.start() - 12)
                context = line[start_col:].strip()
                entry = {"file": rel, "line": lineno, "context": context}
                (definitions if is_def else references).append(entry)

        self._tag_cache[tag] = {
            "tag": tag, "definitions": definitions,
            "references": references,
            "definition_count": len(definitions),
            "reference_count": len(references),
            "scanned_files": scanned, "root": search_root,
            "method": "word-boundary+definition-heuristics",
        }
        return self._tag_cache[tag]

    def recall_search(self, tag: str) -> Optional[Dict[str, Any]]:
        """复查最近一次检索结果 (语义记忆留痕)"""
        cached = self._tag_cache.get(tag)
        if cached is not None:
            return cached
        mem = self.memory.recall(f"tag:{tag}")
        if mem is None:
            return None
        return {"tag": tag, "summary": mem, "method": "semantic-memory"}

    # ── 权重 / 清单 ─────────────────────────────────────────
    def save_weights(self) -> Dict[str, Any]:
        # 检索器为确定性启发式, 无可学习权重
        return {}

    def build_manifest(self, **extra: Any) -> Dict[str, Any]:
        extra.setdefault("capability", self.CAPABILITY)
        extra.setdefault("default_root", "src")
        extra.setdefault("exts", list(DEFAULT_EXTS))
        extra.setdefault("search_count", self.search_count)
        return super().build_manifest(**extra)

    def stats(self) -> Dict[str, Any]:
        s = super().stats()
        s["search_count"] = self.search_count
        s["default_root"] = "src"
        s["exts"] = list(DEFAULT_EXTS)
        return s


# ── 辅助 ──────────────────────────────────────────────────

def _parse_query(data: Any) -> Tuple[str, Optional[str], Optional[List[str]]]:
    """从事件 data 取 (tag, root, exts)"""
    if isinstance(data, str):
        return data.strip(), None, None
    if isinstance(data, dict):
        tag = data.get("tag")
        if isinstance(tag, str):
            return (tag.strip(), data.get("root"), data.get("exts"))
    return "", None, None


def _is_identifier(tag: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tag))


def _iter_source(root: str, exts: Tuple[str, ...]):
    """递归产源文件: (绝对路径, 相对 root 的路径)"""
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(exts):
                full = os.path.join(dirpath, fn)
                yield full, os.path.relpath(full, root)
