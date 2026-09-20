# 模态面存档（.dfpkg，技术文档）

> 本文档是 README「模态面存档」相关技术细节的完整归档：每个模态面可独立打包为
> `.dfpkg` 存档（遵循 CuteMamen 包格式精神），支持自由导入导出与随用随载热加载。
> `.dfpkg` 是 CuteMamen 插件标准的首例特例；插件标准全文见
> [PLUGIN_STANDARD.md](./PLUGIN_STANDARD.md)。

## .dfpkg：模态面独立存档（v0.7.0）

每个模态面可以独立打包为 `.dfpkg` 存档（遵循 CuteMamen 包格式精神：
单个 tar.gz，内含 `manifest.json` 清单 + `weights/` 权重 + `memory/` 状态），
支持自由导入导出与随用随载热加载：

```python
gpt = CubeGPT(depth=1, dim=16, modalities=["numeric", "text"])

# 导出: 把 text 面打包成独立存档
gpt.export_face("text", "text.dfpkg", author="me", capability="文本语义计算")

# 卸载: 自动先导出 pkg 再从内存移除 (可随时恢复)
gpt.unload_face("text")                    # 默认存到 cache/face_pkgs/text.dfpkg
gpt.list_faces()                           # {'loaded': ['numeric'], 'registered': ['text']}

# 随用随载: 注册后不占内存, step() 用到该模态时现场热加载
gpt.register_face_pkg("cache/face_pkgs/text.dfpkg")
spikes = gpt.step({"text": "hello"})       # 触发热加载, 之后常驻

# 或显式加载 / 导入到另一个模型
gpt2 = CubeGPT(depth=1, dim=16, modalities=["numeric"])
gpt2.import_face("text.dfpkg")             # 权重 + 状态逐位还原
```

`manifest.json` 含模态/深度/dim/单元与参数规模/内存占用，带 `min_core_version`
兼容性检查（内核过旧拒绝加载）。权重与状态逐位可复现：感受野投影由 layer_id
的 crc32 种子重建，小世界连接随 STDP 训练后的真值一起存档。