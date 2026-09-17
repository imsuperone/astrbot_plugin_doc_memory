# 文档记忆助手

📚 AstrBot 零 Embedding 文档记忆插件，WebUI 上传 md / txt / pdf / docx / 酒馆角色卡，按群独立绑定，大模型对话自动引用。

-   📦 项目主页：[https://github.com/imsuperone/xbdoc](https://github.com/imsuperone/xbdoc)
-   🔌 插件 ID：`astrbot_plugin_xbdoc`
-   📌 版本：`v1.1.2`，要求 AstrBot `>=4.16`

---

## 🌟 核心特性

-   ⚡ **三大生效模式**：`system` 强制遵守（文档直灌系统提示词） / `workspace` 模拟工作区（`/workspace/` 沙箱挂载） / `reference` 仅作参考资料（提问时按需检索）。
-   🧠 **零 Embedding 检索**：无需向量模型，中英分词 + TF 加权 + 覆盖率计分，切片词频全缓存，Top-K 注入。
-   🍷 **酒馆原生支持**：SillyTavern 角色卡 / 预设自动解析（PNG 内嵌 `chara/ccv3` 与 JSON），`/doc greeting` 直看开场白。
-   👥 **按群彻底隔离**：会话 Key 归一（`group:` / `private:`），绑定、提示词、屏蔽、模式各群独立。
-   🏷️ **群专属提示词**：无文档也可独立生效；`shield` 清空原人格、`force` 强制唯一系统词，三模式下恰好生效一次。
-   🧹 **解绑即恢复出厂**：`/doc unbind` 留空清空文档、提示词、屏蔽、强制注入，空配置条目彻底删除。
-   🧠 **历史强行遗忘**：`/doc no` 截断此前全部上下文，`off` 恢复。
-   💻 **WebUI 管理台**：文档上传 / 预览 / 下载、群搜索（含适配器主动拉取）、绑定表单、模式直选、深浅主题。
-   📦 **改名无痛迁移**：`astrbot_plugin_doc_memory` / `xbdoc` 旧数据目录自动移动迁移。

---

## 🚀 安装

1.  AstrBot 后台 → **插件** → **从链接安装**，填入：
    
    https://github.com/imsuperone/xbdoc.git
    
2.  重启 AstrBot，插件自动加载（如需 PDF / DOCX 解析会自动安装 `requirements.txt` 依赖）。
3.  聊天发送 `/doc` 查看菜单，WebUI 打开 `文档记忆助手` 页面传文档、绑群。

## 🚀 聊天指令（`/doc` 群内管理，点开分组查看）

<details>
<summary>📖 查看</summary>

| 指令 | 说明 |
| --- | --- |
| `/doc` | 完整菜单 |
| `/doc status` | 本群绑定、模式、屏蔽、提示词状态 |
| `/doc list` | 知识库文档与 ID |
| `/doc workspace` | 工作区挂载清单 |
| `/doc search <词>` | 检索绑定文档 |
| `/doc read <ID> [n]` | 预览文档切片 |
| `/doc greeting` | 酒馆角色开场白 |

</details>

<details>
<summary>🔗 绑定（管理员）</summary>

| 指令 | 说明 |
| --- | --- |
| `/doc bind <ID...>` | 追加绑定，自动合并 |
| `/doc unbind [ID...]` | 解绑；留空全清，提示词一并清除 |

</details>

<details>
<summary>🎛️ 模式（管理员，需先绑定文档）</summary>

| 指令 | 说明 |
| --- | --- |
| `/doc mode s\|w\|r` | 强制遵守 / 工作区 / 仅参考 |
| `/doc shield on\|off` | 清空 / 保留原人格 |
| `/doc force on\|off` | 专属提示词唯一生效 |

</details>

<details>
<summary>🏷️ 提示词（管理员）</summary>

| 指令 | 说明 |
| --- | --- |
| `/doc prompt` | 查看本群配置详情 |
| `/doc prompt_set <内容>` | 设置专属提示词 |
| `/doc prompt_clear` | 清除专属提示词 |

</details>

<details>
<summary>🧹 历史（管理员）</summary>

| 指令 | 说明 |
| --- | --- |
| `/doc no [off]` | 忘掉此前消息 / 恢复 |

</details>

---

## 📦 依赖

-   `pypdf >= 4.3.0`、`python-docx >= 1.1.2`（仅 PDF / DOCX 需要，纯文本、Markdown、JSON、酒馆卡开箱即用）
-   上传上限 50MB；切片长度 / 重叠 / Top-K / 注入上限均可在插件配置中调整

## 🧾 备注

-   私聊独立绑定受 `allow_private_bind` 控制（默认开启），WebUI 群列表仅展示群聊。
-   关闭 `auto_inject` 后不再自动检索文档，但专属提示词与屏蔽依然生效。
-   数据持久化在 `data/plugin_data/astrbot_plugin_xbdoc`（`index.json` / `bindings.json` / `seen_groups.json` / `docs/`）。
-   卸载重装不会丢数据；要彻底清零请先停服再删除上述数据目录。
