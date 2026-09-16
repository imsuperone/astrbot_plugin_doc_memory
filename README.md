# 文档记忆助手 xbdoc（`astrbot_plugin_xbdoc`）

让 AI 读懂你的 md / txt / pdf / docx / png / json 文档与酒馆角色卡，并按群记住与执行。

- **作者**：Light
- **开源仓库**：https://github.com/imsuperone/xbdoc
- **当前版本**：1.1.0
- **AstrBot 版本要求**：>=4.16

## 核心特性

- **三大生效模式**：
  - ⚡ **强制遵守模式 (`system`)**：文档直接作为系统提示词载入大模型。
  - 💻 **模拟工作区模式 (`workspace`)**：挂载工作区沙箱，大模型严格基于工作区文件分析回答。
  - 📖 **仅作参考资料模式 (`reference`)**：将文档存入记忆库，对话时按需检索相关片段引用。
- **强制注入系统提示词**：一键清空其他所有提示词，将群专属提示词强制设为唯一 System Prompt。
- **无文档时独立生效**：即使没有绑定任何文档，依然可以独立设置专属系统提示词与人格屏蔽。
- **酒馆 (SillyTavern) 角色卡原生支持**：直接拖拽上传 `.png`（内嵌元数据）与 `.json` 角色卡/预设，可用 `/doc greeting` 查看开场白。
- **历史强行遗忘与清空**：发送 `/doc no` 彻底遗忘此前所有聊天记录。
- **Android 16 管理台**：大圆角药丸卡片设计，支持模式直选与深浅主题切换。

## 安装

把 `astrbot_plugin_xbdoc` 打包为 zip，在 AstrBot WebUI → **插件** → **安装插件** → **上传安装** 即可。

依赖说明：如需解析 PDF 或 DOCX，请安装 `requirements.txt` 中的可选依赖。纯文本、Markdown、JSON 与酒馆 PNG 角色卡无需额外依赖，开箱即用。

## 常用指令

- `/doc`：查看完整指令菜单
- `/doc list`：查看知识库文档列表
- `/doc status`：查看本群绑定状态与生效配置
- `/doc bind <文档ID>`：绑定文档到本群（管理员）
- `/doc unbind [文档ID]`：解绑文档，留空清空本群绑定（管理员）
- `/doc mode workspace|system|reference`：切换生效模式（管理员）
- `/doc workspace`：查看工作区挂载清单
- `/doc force on|off`：切换强制注入系统提示词开关（管理员）
- `/doc shield on|off`：切换人格屏蔽开关（管理员）
- `/doc no [off]`：彻底清空并遗忘此指令之前的历史消息
- `/doc greeting`：查看绑定的酒馆角色开场白
- `/doc search <关键词>`：检索当前群绑定的文档内容
- `/doc prompt_set <内容>`：设置本群专属提示词（管理员）
- `/doc prompt_clear`：清空本群专属提示词（管理员）
