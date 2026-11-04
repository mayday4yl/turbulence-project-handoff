# 主文档生成与渲染记录（准备阶段）

当前主稿为 `../MAIN_HANDOFF_DRAFT.md`，生成器为 `build_handoff_doc.py`。9页 DOCX/PDF 及全部页面 PNG 位于本目录和 `rendered/`。`DRAFT_VISUAL_QA.json` 绑定实际 DOCX、PDF 和9张 PNG 的 SHA；root 已逐页检查。这仅验收准备稿版式，最终数据补齐后必须重新生成、渲染并重新目视全部最终页面。

使用 Codex bundled Python 和 documents skill 的 `render_docx.py`。本机默认 fontconfig 曾令中文缺字，修复限于此目录的 `fonts.conf`：加载 `/Library/Fonts` 中已安装的 Arial Unicode MS，并把字体缓存置于 `/private/tmp/unidb-handoff-fontcache`。没有修改系统字体或训练环境。失败准备稿保留在 `render_font_fail/` 及带 `font_fail` 后缀的 DOCX。

实际生成命令（当前仅准备稿）：

```sh
'/Users/lucas/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' '/Users/lucas/Desktop/turbulence IR/project_handoff_20261104/preparation/document_build/build_handoff_doc.py'
```

实际渲染命令：

```sh
FONTCONFIG_FILE='/Users/lucas/Desktop/turbulence IR/project_handoff_20261104/preparation/document_build/fonts.conf' '/Users/lucas/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' '/Users/lucas/.codex/plugins/cache/openai-primary-runtime/documents/26.1026.10807/skills/documents/render_docx.py' '/Users/lucas/Desktop/turbulence IR/project_handoff_20261104/preparation/document_build/Turbulence_Project_Handoff_DRAFT.docx' --output_dir '/Users/lucas/Desktop/turbulence IR/project_handoff_20261104/preparation/document_build/rendered' --emit_pdf
```

最终模式当前拒绝执行。实验最终完整验收、实际 best/latest 备份、最终曲线和固定图、交接资料包与链接验收完成前，不得把此稿改名当最终交付。最终编辑还需加入真实最终状态、正式文件链接和接收清单；准备稿的通过不能代替最终验收。
