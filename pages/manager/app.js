// Document Memory Assistant - Android 16 (Material 3 Expressive) Client
(function () {
  "use strict";

  const PLUGIN_ID = "astrbot_plugin_xbdoc";

  // ---- Safe Bridge & API Layer ----
  function getBridge() {
    try {
      if (window.AstrBotPluginPage && typeof window.AstrBotPluginPage.apiGet === "function") {
        return window.AstrBotPluginPage;
      }
      if (window.bridge && typeof window.bridge.apiGet === "function") {
        return window.bridge;
      }
      if (window.parent && window.parent.AstrBotPluginPage && typeof window.parent.AstrBotPluginPage.apiGet === "function") {
        return window.parent.AstrBotPluginPage;
      }
      if (window.parent && window.parent.bridge && typeof window.parent.bridge.apiGet === "function") {
        return window.parent.bridge;
      }
    } catch (e) {
      console.warn("[DocMemory] getBridge cross-frame check ignored:", e);
    }
    return null;
  }

  let _detectedPrefix = null;

  async function tryFetchJson(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${errText || res.statusText}`);
    }
    return await res.json();
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const res = String(reader.result || "");
        resolve(res.includes(",") ? res.split(",")[1] : res);
      };
      reader.onerror = (e) => reject(e);
      reader.readAsDataURL(file);
    });
  }

  const api = {
    async ready() {
      const b = getBridge();
      if (b && typeof b.ready === "function") {
        try {
          await b.ready();
        } catch (e) {
          console.warn("[DocMemory] bridge.ready error:", e);
        }
      }
    },

    async get(endpoint, params = {}) {
      const cleanParams = { ...params };
      const b = getBridge();
      if (b && typeof b.apiGet === "function") {
        return await b.apiGet(endpoint, cleanParams);
      }

      const qs = new URLSearchParams(cleanParams).toString();
      const queryStr = qs ? `?${qs}` : "";

      if (_detectedPrefix) {
        try {
          return await tryFetchJson(`${_detectedPrefix}${endpoint}${queryStr}`);
        } catch (e) {}
      }

      const prefixes = [
        `/${PLUGIN_ID}/`,
        `/api/plugins/${PLUGIN_ID}/`,
        `api/`,
        `./api/`,
        `./`,
      ];

      for (const p of prefixes) {
        try {
          const res = await tryFetchJson(`${p}${endpoint}${queryStr}`);
          _detectedPrefix = p;
          return res;
        } catch (e) {}
      }

      throw new Error(`无法连接至插件后端 API (${endpoint})`);
    },

    async post(endpoint, data = {}) {
      const b = getBridge();
      if (b && typeof b.apiPost === "function") {
        return await b.apiPost(endpoint, data);
      }

      const options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      };

      if (_detectedPrefix) {
        try {
          return await tryFetchJson(`${_detectedPrefix}${endpoint}`, options);
        } catch (e) {}
      }

      const prefixes = [
        `/${PLUGIN_ID}/`,
        `/api/plugins/${PLUGIN_ID}/`,
        `api/`,
        `./api/`,
        `./`,
      ];

      for (const p of prefixes) {
        try {
          const res = await tryFetchJson(`${p}${endpoint}`, options);
          _detectedPrefix = p;
          return res;
        } catch (e) {}
      }

      throw new Error(`请求后端失败 (${endpoint})`);
    },

    async upload(endpoint, file) {
      // 1. 优先使用 base64 JSON 直传（彻底消除 iframe 跨域 FormData 克隆失效与字段名不匹配问题）
      try {
        const b64 = await fileToBase64(file);
        if (b64) {
          return await api.post(endpoint, {
            filename: file.name,
            file_base64: b64,
          });
        }
      } catch (e) {
        console.warn("[DocMemory] Base64 upload fallback:", e);
      }

      // 2. 备用方式：FormData
      const formData = new FormData();
      formData.append("file", file);

      const b = getBridge();
      if (b && typeof b.upload === "function") {
        try {
          return await b.upload(endpoint, file);
        } catch (e) {}
      }

      const prefixes = [
        _detectedPrefix || `/${PLUGIN_ID}/`,
        `/${PLUGIN_ID}/`,
        `/api/plugins/${PLUGIN_ID}/`,
      ];

      for (const pfx of prefixes) {
        try {
          const res = await fetch(`${pfx}${endpoint}`, {
            method: "POST",
            body: formData,
          });
          if (res.ok) {
            _detectedPrefix = pfx;
            return await res.json();
          }
        } catch (e) {}
      }

      throw new Error("上传请求未成功发送");
    },

    async download(endpoint, params = {}) {
      const b = getBridge();
      if (b && typeof b.download === "function") {
        return await b.download(endpoint, params);
      }

      const qs = new URLSearchParams(params).toString();
      const pfx = _detectedPrefix || `/${PLUGIN_ID}/`;
      const url = `${pfx}${endpoint}${qs ? "?" + qs : ""}`;

      const link = document.createElement("a");
      link.href = url;
      link.download = "";
      link.target = "_blank";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    },
  };

  // ---- Safe In-Memory Storage (Prevents Sandboxed Iframe Exceptions) ----
  const memoryStore = {};
  function safeGet(key, fallback = null) {
    try {
      if (window.localStorage) {
        const val = window.localStorage.getItem(key);
        return val !== null ? val : fallback;
      }
    } catch (e) {}
    return memoryStore[key] !== undefined ? memoryStore[key] : fallback;
  }

  function safeSet(key, val) {
    memoryStore[key] = val;
    try {
      if (window.localStorage) {
        window.localStorage.setItem(key, val);
      }
    } catch (e) {}
  }

  // ---- DOM Helper ----
  const $ = (id) => document.getElementById(id);

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  // ---- Toast / Snackbar Notification ----
  let toastTimer = null;
  function showToast(msg, duration = 3000) {
    const toast = $("snackbar");
    const text = $("snackbarText");
    if (!toast || !text) return;

    text.textContent = msg;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove("show");
    }, duration);
  }

  // ---- State Management ----
  let docsList = [];
  let bindingsMap = {};
  let groupsCache = [];
  let selectedDocIds = new Set();
  let currentReaderDoc = null;
  let currentReaderChunk = 1;
  let currentReaderTotal = 1;
  let currentShield = "off";
  let currentDocMode = "reference";
  let currentForcePrompt = false;
  let currentBindingFilter = "all";

  // ---- Theme Handling ----
  function initTheme() {
    try {
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      const initial = safeGet("doc_memory_theme", prefersDark ? "dark" : "light");
      applyTheme(initial);

      const btn = $("themeToggleBtn");
      if (btn) {
        btn.addEventListener("click", () => {
          const cur = document.documentElement.getAttribute("data-theme") || "light";
          const next = cur === "dark" ? "light" : "dark";
          applyTheme(next);
        });
      }
    } catch (e) {
      console.warn("[DocMemory] initTheme failed:", e);
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    safeSet("doc_memory_theme", theme);
    const icon = $("themeIcon");
    if (!icon) return;
    if (theme === "dark") {
      icon.innerHTML = `<path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37c-.39-.39-1.03-.39-1.41 0s-.39 1.03 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41l-1.06-1.06zm1.06-10.96c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.36c.39-.39.39-1.03 0-1.41s-1.03-.39-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z"/>`;
    } else {
      icon.innerHTML = `<path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9c0-.46-.04-.92-.1-1.36-.98 1.37-2.58 2.26-4.4 2.26-2.98 0-5.4-2.42-5.4-5.4 0-1.81.89-3.42 2.26-4.4-.44-.06-.9-.1-1.36-.1z"/>`;
    }
  }

  // ---- Navigation Tabs ----
  function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        const targetId = tab.dataset.tab;
        const target = $(targetId);
        if (target) target.classList.add("active");
      });
    });
  }

  function switchTab(tabId) {
    const btn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
    if (btn) btn.click();
  }

  // ---- Stats Counters ----
  function updateStats() {
    const docEl = $("statDocs");
    const chunkEl = $("statChunks");
    const bindEl = $("statBindings");

    if (docEl) docEl.textContent = docsList.length;
    if (chunkEl) {
      const total = docsList.reduce((acc, d) => acc + (parseInt(d.chunks) || 0), 0);
      chunkEl.textContent = total;
    }
    if (bindEl) bindEl.textContent = Object.keys(bindingsMap).length;
  }

  // ---- Load Docs ----
  async function loadDocs() {
    try {
      const res = await api.get("docs");
      docsList = res.docs || res.data?.docs || [];
      renderDocs(docsList);
      renderDocChips();
      updateStats();
    } catch (e) {
      console.error("[DocMemory] loadDocs error:", e);
      renderDocs([]);
      showToast("获取文档失败: " + e.message);
    }
  }

  // ---- Load Bindings ----
  async function loadBindings() {
    try {
      const res = await api.get("bindings");
      bindingsMap = res.bindings || {};
      renderBindings();
      updateStats();
    } catch (e) {
      console.error("[DocMemory] loadBindings error:", e);
      renderBindings();
      showToast("获取绑定失败: " + e.message);
    }
  }

  // ---- Render Document Cards with Event Delegation ----
  function renderDocs(list) {
    const container = $("docGrid");
    if (!container) return;

    if (!list.length) {
      container.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-state-icon">📂</div>
          <h3>暂无入库文档</h3>
          <p>支持将 .md / .txt / .pdf / .docx 文件拖放至上方卡片直接上传</p>
        </div>`;
      return;
    }

    container.innerHTML = list.map((doc) => {
      const suffix = (doc.suffix || "").replace(".", "").toLowerCase();
      const isTavern = Boolean(doc.is_tavern || String(doc.filename || "").includes("酒馆"));
      let typeClass = "txt";
      let typeLabel = (suffix || "TXT").toUpperCase();
      if (isTavern) {
        typeClass = "tavern";
        typeLabel = "🍷 酒馆";
      } else if (suffix === "md" || suffix === "markdown") {
        typeClass = "md";
      } else if (suffix === "pdf") {
        typeClass = "pdf";
      } else if (suffix === "docx") {
        typeClass = "docx";
      }

      return `
        <div class="doc-card" data-id="${esc(doc.doc_id)}">
          <div class="doc-card-top">
            <div class="doc-type-icon ${typeClass}">${esc(typeLabel)}</div>
            <div class="doc-card-info">
              <div class="doc-card-title" title="${esc(doc.filename)}">${esc(doc.filename)}</div>
              <div class="doc-card-badges">
                ${isTavern ? '<span class="badge-pill" style="background:#fce7f3; color:#9d174d; font-weight:700;">🍷 酒馆角色卡</span>' : ''}
                <span class="badge-pill id-badge">ID: ${esc(doc.doc_id)}</span>
                <span class="badge-pill">${esc(doc.chunks)} 切片</span>
                <span class="badge-pill">${esc(doc.text_len)} 字</span>
              </div>
            </div>
          </div>
          <div class="doc-card-actions">
            <button class="m3-btn m3-btn-outlined m3-btn-sm" data-act="preview" data-id="${esc(doc.doc_id)}" type="button">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>
              预览
            </button>
            <button class="m3-btn m3-btn-tonal m3-btn-sm" data-act="download" data-id="${esc(doc.doc_id)}" type="button">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/></svg>
              下载
            </button>
            <button class="m3-btn m3-btn-tonal m3-btn-sm" data-act="attach" data-id="${esc(doc.doc_id)}" type="button">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z"/></svg>
              绑定
            </button>
            <button class="m3-btn m3-btn-danger m3-btn-sm" data-act="delete" data-id="${esc(doc.doc_id)}" title="删除文档" type="button">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>
            </button>
          </div>
        </div>
      `;
    }).join("");
  }

  // Attach Document Grid Click Delegation (Runs once, never drops events)
  function initDocGridEvents() {
    const container = $("docGrid");
    if (!container) return;

    container.addEventListener("click", async (e) => {
      const badge = e.target.closest(".id-badge");
      if (badge) {
        const card = badge.closest(".doc-card");
        const did = card ? card.dataset.id : "";
        if (did) {
          try {
            await navigator.clipboard.writeText(did);
            showToast(`✅ 已复制文档 ID: ${did}`);
          } catch {
            showToast(`文档 ID: ${did}`);
          }
        }
        return;
      }

      const btn = e.target.closest("[data-act]");
      if (!btn) return;
      e.stopPropagation();

      const act = btn.dataset.act;
      const id = btn.dataset.id;
      if (!id) return;

      if (act === "preview") {
        openReader(id, 1);
      } else if (act === "download") {
        try {
          showToast("开始下载文档…");
          await api.download("docs/download", { doc_id: id });
        } catch (err) {
          showToast("下载失败: " + err.message);
        }
      } else if (act === "attach") {
        selectedDocIds.add(id);
        renderDocChips();
        switchTab("tab-bindings");
        const sk = $("sessionKey");
        if (sk) sk.focus();
        showToast("已添加该文档至绑定表单");
      } else if (act === "delete") {
        try {
          showToast(`正在删除文档 (ID: ${id})…`);
          await api.post("docs/delete", { doc_id: id });
          showToast("✅ 文档已成功删除，相关绑定已同步清理");
          await loadDocs();
          await loadBindings();
        } catch (err) {
          showToast("删除失败: " + err.message);
        }
      }
    });
  }

  // ---- Document Search Filter ----
  function initDocSearch() {
    const input = $("docSearchInput");
    if (!input) return;

    input.addEventListener("input", (e) => {
      const q = e.target.value.trim().toLowerCase();
      if (!q) {
        renderDocs(docsList);
        return;
      }
      const filtered = docsList.filter((d) =>
        (d.filename || "").toLowerCase().includes(q) || (d.doc_id || "").toLowerCase().includes(q)
      );
      renderDocs(filtered);
    });
  }

  // ---- File Upload & Drag & Drop ----
  function initDropzone() {
    const fileInput = $("fileInput");
    const dropzone = $("dropzone");
    const progress = $("uploadProgress");

    if (!fileInput || !dropzone) return;

    // File selected
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) handleUpload(file);
      fileInput.value = "";
    });

    // Drag events
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });

    ["dragleave", "dragend"].forEach((type) => {
      dropzone.addEventListener(type, () => dropzone.classList.remove("drag-over"));
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
      const file = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files[0] : null;
      if (file) handleUpload(file);
    });

    async function handleUpload(file) {
      if (!file) return;
      if (file.size > 50 * 1024 * 1024) {
        showToast("文件超出 50MB 上限，请拆分后上传");
        return;
      }
      if (progress) progress.style.display = "block";
      showToast(`正在上传并切片 ${file.name}…`, 5000);

      try {
        const res = await api.upload("docs/upload", file);
        if (progress) progress.style.display = "none";
        showToast(`✅ 文档 ${res.doc?.filename || file.name} 入库成功！`);
        await loadDocs();
      } catch (err) {
        if (progress) progress.style.display = "none";
        showToast("❌ 上传失败: " + err.message);
      }
    }
  }

  // ---- Document Chips Picker (Bindings View) ----
  function renderDocChips() {
    const container = $("docChipsPicker");
    if (!container) return;

    if (!docsList.length) {
      container.innerHTML = `<span class="helper">知识库暂无文档，请先在文档库上传。</span>`;
      return;
    }

    container.innerHTML = docsList.map((d) => {
      const isSelected = selectedDocIds.has(d.doc_id);
      return `
        <div class="doc-select-chip ${isSelected ? "selected" : ""}" data-id="${esc(d.doc_id)}" role="button" tabindex="0">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="${isSelected ? 'M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' : 'M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z'}"/>
          </svg>
          <span>${esc(d.filename)}</span>
        </div>
      `;
    }).join("");

    const bindIdsInput = $("bindIds");
    if (bindIdsInput) {
      bindIdsInput.value = Array.from(selectedDocIds).join(",");
    }

    const hasDocs = selectedDocIds.size > 0;
    const modeRow = $("docModeRow");
    const modeSub = $("docModeSub");
    if (modeRow) {
      modeRow.classList.remove("disabled");
      modeRow.querySelectorAll("button").forEach((b) => b.disabled = false);
      if (modeSub) modeSub.textContent = hasDocs
        ? "设置文档在会话中的角色定位与隔离级别"
        : "当前未选择文档也可预设模式，绑定文档后自动生效";
    }
  }

  function initDocChipsEvents() {
    const container = $("docChipsPicker");
    if (!container) return;

    container.addEventListener("click", (e) => {
      const chip = e.target.closest(".doc-select-chip");
      if (!chip) return;
      const id = chip.dataset.id;
      if (!id) return;

      if (selectedDocIds.has(id)) {
        selectedDocIds.delete(id);
      } else {
        selectedDocIds.add(id);
      }
      renderDocChips();
    });
  }

  // ---- Group Search Auto-Suggest ----
  let suggestTimer = null;
  async function searchGroups(q = "") {
    try {
      const res = await api.get("groups", q ? { q, limit: 30 } : { limit: 30 });
      groupsCache = res.groups || res.data?.groups || [];
      return groupsCache;
    } catch (e) {
      return [];
    }
  }

  function renderGroupSuggest(list) {
    const box = $("suggestDropdown");
    if (!box) return;

    if (!list.length) {
      box.classList.remove("open");
      box.innerHTML = "";
      return;
    }

    box.innerHTML = list.slice(0, 15).map((g) => {
      const firstLetter = (g.group_name || g.gid || "群").charAt(0).toUpperCase();
      return `
        <div class="suggest-card" data-key="${esc(g.session_key || ('group:' + g.gid))}">
          <div style="display:flex; align-items:center;">
            <div class="suggest-avatar">${esc(firstLetter)}</div>
            <div class="suggest-info">
              <strong>${esc(g.group_name || ("群聊 " + g.gid))}</strong>
              <span>${esc(g.platform ? g.platform + ' · ' : '')}群号: ${esc(g.gid)} ${g.msg_count ? ' · ' + g.msg_count + '条发言' : ''}</span>
            </div>
          </div>
          ${g.bound ? '<span class="badge-pill" style="background:var(--md-sys-color-primary-container); color:var(--md-sys-color-primary); font-weight:600;">已绑定</span>' : '<span class="badge-pill">选用</span>'}
        </div>
      `;
    }).join("");

    box.classList.add("open");
  }

  function initGroupSuggest() {
    const input = $("sessionKey");
    const box = $("suggestDropdown");
    const fetchBtn = $("fetchBotGroupsBtn");
    if (!input || !box) return;

    if (fetchBtn) {
      fetchBtn.addEventListener("click", async () => {
        const originalHtml = fetchBtn.innerHTML;
        fetchBtn.disabled = true;
        fetchBtn.innerHTML = `<span>拉取中…</span>`;
        showToast("正在向机器人适配器拉取已加入的群聊…", 4000);

        try {
          let res = null;
          try {
            res = await api.post("groups/fetch");
          } catch (e) {
            res = await api.get("groups", { refresh: "1" });
          }

          const list = res.groups || res.data?.groups || [];
          const newCount = res.new_fetched !== undefined ? res.new_fetched : list.length;
          groupsCache = list.length ? list : await searchGroups("");
          renderGroupSuggest(groupsCache);
          input.focus();

          if (newCount > 0) {
            showToast(`✅ 成功从适配器拉取到 ${newCount} 个群聊（共收录 ${groupsCache.length} 个）！`, 3500);
          } else if (groupsCache.length > 0) {
            showToast(`💡 适配器未返回新群，已为您列出 ${groupsCache.length} 个已知群聊`, 4000);
          } else {
            showToast(`💡 提示：当前协议若不支持拉取群清单（如QQ官方Bot/Telegram/Discord），在群里发一条消息机器人即可自动记录群号，或直接手填 group:群号。`, 6000);
          }
        } catch (err) {
          showToast("获取群聊列表失败: " + err.message);
        } finally {
          fetchBtn.disabled = false;
          fetchBtn.innerHTML = originalHtml;
        }
      });
    }

    box.addEventListener("click", (e) => {
      const card = e.target.closest(".suggest-card");
      if (!card) return;
      const key = card.dataset.key;
      input.value = key;
      box.classList.remove("open");
      loadExistingSessionSettings(key);
    });

    input.addEventListener("input", () => {
      clearTimeout(suggestTimer);
      const q = input.value.trim();
      if (!q) {
        searchGroups("").then(renderGroupSuggest);
        return;
      }
      if (/^group:\d+$/.test(q)) {
        box.classList.remove("open");
        return;
      }
      suggestTimer = setTimeout(async () => {
        const list = await searchGroups(q);
        renderGroupSuggest(list);
      }, 220);
    });

    const showList = async () => {
      const q = input.value.trim();
      if (/^group:\d+$/.test(q)) return;
      const list = groupsCache.length ? groupsCache : await searchGroups(q);
      renderGroupSuggest(list);
    };

    input.addEventListener("focus", showList);
    input.addEventListener("click", showList);

    document.addEventListener("click", (e) => {
      if (!e.target.closest("#sessionKey") && !e.target.closest("#suggestDropdown") && !e.target.closest("#fetchBotGroupsBtn")) {
        box.classList.remove("open");
      }
    });
  }

  function loadExistingSessionSettings(key) {
    const entry = bindingsMap[key];
    if (!entry) return;

    selectedDocIds.clear();
    const docs = entry.docs || [];
    docs.forEach((d) => selectedDocIds.add(typeof d === "string" ? d : d.doc_id));
    renderDocChips();

    const promptEl = $("bindPrompt");
    if (promptEl) promptEl.value = entry.prompt || "";

    const shieldVal = entry.shield ? "on" : "off";
    setShieldChoice(shieldVal);

    const forceVal = entry.force_system_prompt ? "on" : "off";
    setForceChoice(forceVal);

    let modeVal = "reference";
    if (entry.mode === "system" || entry.mode === "workspace") {
      modeVal = entry.mode;
    }
    setDocMode(modeVal);
  }

  // ---- Document Execution Mode Choice Buttons ----
  function initDocMode() {
    const group = $("docModeGroup");
    if (!group) return;

    group.addEventListener("click", (e) => {
      const btn = e.target.closest(".segmented-choice-btn");
      if (!btn) return;
      setDocMode(btn.dataset.val);
    });
  }

  function setDocMode(val) {
    if (val === "system" || val === "workspace") {
      currentDocMode = val;
    } else {
      currentDocMode = "reference";
    }
    const group = $("docModeGroup");
    if (!group) return;

    group.querySelectorAll(".segmented-choice-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.val === currentDocMode);
    });
  }

  // ---- Force System Prompt Choice Buttons ----
  function initForceChoice() {
    const group = $("forceChoiceGroup");
    if (!group) return;

    group.addEventListener("click", (e) => {
      const btn = e.target.closest(".segmented-choice-btn");
      if (!btn) return;
      setForceChoice(btn.dataset.val);
    });
  }

  function setForceChoice(val) {
    currentForcePrompt = val === "on";
    const group = $("forceChoiceGroup");
    if (!group) return;

    group.querySelectorAll(".segmented-choice-btn").forEach((btn) => {
      btn.classList.toggle("active", (btn.dataset.val === "on") === currentForcePrompt);
    });
  }

  // ---- Shield Switch Choice Buttons (Only On / Off, No Global) ----
  function initShieldChoice() {
    const group = $("shieldChoiceGroup");
    if (!group) return;

    group.addEventListener("click", (e) => {
      const btn = e.target.closest(".segmented-choice-btn");
      if (!btn) return;
      setShieldChoice(btn.dataset.val);
    });
  }

  function setShieldChoice(val) {
    currentShield = val === "on" ? "on" : "off";
    const group = $("shieldChoiceGroup");
    if (!group) return;

    group.querySelectorAll(".segmented-choice-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.val === currentShield);
    });
  }

  // ---- Binding Form Save & Reset ----
  function initBindingForm() {
    const resetBtn = $("resetBindFormBtn");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        const sk = $("sessionKey");
        const bp = $("bindPrompt");
        if (sk) sk.value = "";
        if (bp) bp.value = "";
        selectedDocIds.clear();
        renderDocChips();
        setShieldChoice("off");
        setForceChoice("off");
        setDocMode("reference");
        showToast("已清空表单输入");
      });
    }

    const saveBtn = $("saveBindBtn");
    if (saveBtn) {
      saveBtn.addEventListener("click", async () => {
        const input = $("sessionKey");
        let key = input ? input.value.trim() : "";
        if (!key) {
          showToast("请填写或选择目标会话 Key (例如 group:123456)");
          if (input) input.focus();
          return;
        }

        if (/^\d{5,}$/.test(key)) {
          key = "group:" + key;
          input.value = key;
        }

        const ids = Array.from(selectedDocIds);
        const promptEl = $("bindPrompt");
        const prompt = promptEl ? promptEl.value.trim() : "";
        const shield = currentShield === "on";
        const mode = currentDocMode || "reference";
        const forceSys = currentForcePrompt;

        try {
          await api.post("bindings/save", {
            session_key: key,
            doc_ids: ids,
            prompt: prompt,
            shield: shield,
            mode: mode,
            force_system_prompt: forceSys,
          });
          showToast("✅ 会话绑定与配置已成功保存！");
          await loadBindings();
          await searchGroups("");
        } catch (err) {
          showToast("❌ 保存失败: " + err.message);
        }
      });
    }
  }

  // ---- Render Active Bindings List ----
  function renderBindings() {
    const container = $("bindingList");
    if (!container) return;

    const allKeys = Object.keys(bindingsMap);

    // Compute Shield counts (Only On vs Off)
    let onCount = 0;
    let offCount = 0;

    allKeys.forEach((k) => {
      if (Boolean((bindingsMap[k] || {}).shield)) onCount++;
      else offCount++;
    });

    const cAll = $("countAll");
    const cOn = $("countShieldOn");
    const cOff = $("countShieldOff");
    if (cAll) cAll.textContent = allKeys.length;
    if (cOn) cOn.textContent = onCount;
    if (cOff) cOff.textContent = offCount;

    // Filter keys
    const filteredKeys = allKeys.filter((k) => {
      const sh = Boolean((bindingsMap[k] || {}).shield);
      if (currentBindingFilter === "shield_on") return sh === true;
      if (currentBindingFilter === "shield_off") return sh === false;
      return true;
    });

    if (!allKeys.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🔗</div>
          <h3>暂无已绑定的会话</h3>
          <p>您可以在上方选择群聊与文档进行绑定，也可在群内发送 /doc bind 快捷绑定。</p>
        </div>`;
      return;
    }

    if (!filteredKeys.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">🔍</div>
          <h3>未找到符合筛选条件的会话</h3>
          <p>当前筛选状态下无匹配项，点击上方“全部会话”可查看所有记录。</p>
        </div>`;
      return;
    }

    container.innerHTML = filteredKeys.map((k) => {
      const raw = bindingsMap[k] || {};
      const docs = raw.docs || [];
      const prompt = raw.prompt || "";
      const isShield = Boolean(raw.shield);
      const shieldClass = isShield ? "shield-badge-on" : "shield-badge-off";
      const shieldTag = isShield ? "🛡️ 屏蔽已开启 (清空原人格)" : "👤 屏蔽已关闭 (保留原人格)";

      const curMode = raw.mode === "system" ? "system" : (raw.mode === "workspace" ? "workspace" : "reference");

      // Group Name display
      const groupDisplayName = raw.group_name ? raw.group_name : (raw.gid ? `群聊 ${raw.gid}` : k);

      return `
        <div class="binding-card" data-key="${esc(k)}">
          <div class="binding-card-meta">
            <div class="binding-card-key">
              <span class="binding-group-name">👥 ${esc(groupDisplayName)}</span>
              <span class="badge-pill id-badge">${esc(k)}</span>
            </div>
            <div class="binding-card-docs">
              ${docs.length ? docs.map((d) => `<span class="doc-tag">${esc(d.filename || d.doc_id || d)}</span>`).join("") : '<span class="helper">无绑定文档</span>'}
              ${raw.force_system_prompt ? '<span class="badge-pill" style="background:#fee2e2; color:#991b1b; font-weight:700; border:1px solid #f87171;">⚡ 强制唯一系统词</span>' : ''}
              <span class="badge-pill ${shieldClass}">${esc(shieldTag)}</span>
              ${prompt ? '<span class="badge-pill" style="background:var(--md-sys-color-tertiary-container); color:var(--md-sys-color-on-tertiary-container);">🏷️ 专属提示词</span>' : ''}
            </div>
            <div class="mode-select-row">
              <span style="font-size:12px; font-weight:600; color:var(--md-sys-color-outline); margin-right:4px;">生效模式:</span>
              ${docs.length ? `
                <button class="mode-btn-pill ${curMode === 'reference' ? 'active' : ''}" data-act="set-mode" data-mode="reference" data-key="${esc(k)}" type="button">📖 仅作参考</button>
                <button class="mode-btn-pill ${curMode === 'system' ? 'active' : ''}" data-act="set-mode" data-mode="system" data-key="${esc(k)}" type="button">⚡ 强制系统词</button>
                <button class="mode-btn-pill ${curMode === 'workspace' ? 'active' : ''}" data-act="set-mode" data-mode="workspace" data-key="${esc(k)}" type="button">💻 工作区Agent</button>
              ` : `
                <span class="helper" style="font-size:12px;">（未绑定文档，模式已禁用。仅专属系统词生效）</span>
              `}
            </div>
          </div>
          <div class="binding-card-actions">
            <button class="m3-btn m3-btn-tonal m3-btn-sm" data-act="edit" data-key="${esc(k)}" type="button">载入编辑</button>
            <button class="m3-btn m3-btn-danger m3-btn-sm" data-act="unbind" data-key="${esc(k)}" type="button">解绑</button>
          </div>
        </div>
      `;
    }).join("");
  }

  // ---- Shield Filter Tabs Handler ----
  function initShieldFilter() {
    const group = $("shieldFilterGroup");
    if (!group) return;

    group.addEventListener("click", (e) => {
      const chip = e.target.closest(".shield-filter-chip");
      if (!chip) return;
      group.querySelectorAll(".shield-filter-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      currentBindingFilter = chip.dataset.filter || "all";
      renderBindings();
    });
  }

  function initBindingListEvents() {
    const container = $("bindingList");
    if (!container) return;

    container.addEventListener("click", async (e) => {
      const idBadge = e.target.closest(".id-badge");
      if (idBadge) {
        const card = idBadge.closest(".binding-card");
        const sk = card ? card.dataset.key : "";
        if (sk) {
          try {
            await navigator.clipboard.writeText(sk);
            showToast(`✅ 已复制会话 Key: ${sk}`);
          } catch {
            showToast(`会话 Key: ${sk}`);
          }
        }
        return;
      }

      const btn = e.target.closest("[data-act]");
      if (!btn) return;
      const act = btn.dataset.act;
      const key = btn.dataset.key;
      if (!key) return;

      if (act === "set-mode") {
        const targetMode = btn.dataset.mode;
        const cur = bindingsMap[key] || {};
        if (cur.mode === targetMode) return;

        const modeLabels = {
          system: "⚡ 已切换为【强制遵守文档（系统提示词）】！",
          workspace: "💻 已切换为【模拟工作区 Agent 模式】！",
          reference: "📖 已切换为【仅作参考资料（记忆库）】！",
        };

        try {
          await api.post("bindings/save", {
            session_key: key,
            doc_ids: (cur.docs || []).map((d) => (typeof d === "string" ? d : d.doc_id)),
            prompt: cur.prompt || "",
            shield: Boolean(cur.shield),
            force_system_prompt: Boolean(cur.force_system_prompt),
            mode: targetMode,
          });
          showToast(modeLabels[targetMode] || "模式已更新");
          await loadBindings();
        } catch (err) {
          showToast("模式切换失败: " + err.message);
        }
        return;
      }

      if (act === "edit") {
        const input = $("sessionKey");
        if (input) {
          input.value = key;
          loadExistingSessionSettings(key);
          window.scrollTo({ top: 180, behavior: "smooth" });
          showToast("已载入会话配置");
        }
      } else if (act === "unbind") {
        try {
          showToast(`正在解绑 ${key} 的文档与相关配置…`);
          await api.post("bindings/save", {
            session_key: key,
            doc_ids: [],
            prompt: "",
            shield: false,
            force_system_prompt: false,
            mode: "reference",
          });
          // 如果当前输入框恰好载入了该群，同步清空表单已勾选文档并禁用模式
          const curInputKey = ($("sessionKey")?.value || "").trim();
          if (curInputKey === key) {
            selectedDocIds.clear();
            renderDocChips();
          }
          showToast(`✅ 已成功解除 ${key} 的全部文档绑定`);
          await loadBindings();
        } catch (err) {
          showToast("解绑失败: " + err.message);
        }
      }
    });
  }

  // ---- Bottom Sheet Document Reader ----
  async function openReader(docId, chunk = 1) {
    currentReaderDoc = docId;
    currentReaderChunk = chunk;
    const modal = $("readerModal");
    if (!modal) return;

    modal.classList.add("open");

    const titleEl = $("readerTitle");
    const subEl = $("readerSub");
    const textEl = $("readerText");
    const prevBtn = $("prevChunkBtn");
    const nextBtn = $("nextChunkBtn");

    if (titleEl) titleEl.textContent = "正在加载切片…";
    if (subEl) subEl.textContent = `切片 ${chunk}`;
    if (textEl) textEl.textContent = "正在向服务器请求文档片段…";

    try {
      const res = await api.get("docs/content", { doc_id: docId, chunk });
      currentReaderTotal = res.total || 1;
      if (titleEl) titleEl.textContent = res.meta?.filename || res.filename || docId;
      if (subEl) subEl.textContent = `切片 ${res.chunk} / ${res.total} (共 ${res.meta?.text_len || 0} 字)`;
      if (textEl) textEl.textContent = res.preview || "（该切片暂无文本内容）";

      if (prevBtn) prevBtn.disabled = currentReaderChunk <= 1;
      if (nextBtn) nextBtn.disabled = currentReaderChunk >= currentReaderTotal;
    } catch (err) {
      if (textEl) textEl.textContent = "读取切片失败: " + err.message;
    }
  }

  function closeReader() {
    const modal = $("readerModal");
    if (modal) modal.classList.remove("open");
  }

  function initReaderEvents() {
    const closeBtn = $("closeReaderBtn");
    if (closeBtn) closeBtn.addEventListener("click", closeReader);

    const modal = $("readerModal");
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) closeReader();
      });
    }

    const prevBtn = $("prevChunkBtn");
    if (prevBtn) {
      prevBtn.addEventListener("click", () => {
        if (currentReaderDoc && currentReaderChunk > 1) {
          openReader(currentReaderDoc, currentReaderChunk - 1);
        }
      });
    }

    const nextBtn = $("nextChunkBtn");
    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        if (currentReaderDoc && currentReaderChunk < currentReaderTotal) {
          openReader(currentReaderDoc, currentReaderChunk + 1);
        }
      });
    }

    const copyBtn = $("copyChunkBtn");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        const textEl = $("readerText");
        const text = textEl ? textEl.textContent : "";
        if (!text) return;
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
          } else {
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.opacity = "0";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }
          showToast("✅ 已复制切片文本到剪贴板");
        } catch (e) {
          showToast("复制失败，请手动选取复制");
        }
      });
    }

    const bindBtn = $("bindFromReaderBtn");
    if (bindBtn) {
      bindBtn.addEventListener("click", () => {
        if (currentReaderDoc) {
          selectedDocIds.add(currentReaderDoc);
          renderDocChips();
          closeReader();
          switchTab("tab-bindings");
          const input = $("sessionKey");
          if (input) input.focus();
          showToast("已选择该文档，请选定绑定的会话");
        }
      });
    }
  }

  // ---- Refresh All Data Button ----
  function initRefreshButton() {
    const btn = $("refreshAllBtn");
    if (!btn) return;

    btn.addEventListener("click", async () => {
      showToast("正在刷新全部数据…");
      try {
        await Promise.all([loadDocs(), loadBindings(), searchGroups("")]);
        showToast("数据已刷新完毕");
      } catch (err) {
        showToast("刷新部分失败: " + err.message);
      }
    });
  }

  // ---- App Startup Entry ----
  async function startApp() {
    console.log("[DocMemory] Starting Android 16 UI application...");

    // 1. Synchronous UI initialization (never blocks)
    initTheme();
    initTabs();
    initDocGridEvents();
    initDocSearch();
    initDropzone();
    initDocChipsEvents();
    initGroupSuggest();
    initDocMode();
    initShieldChoice();
    initForceChoice();
    initShieldFilter();
    initBindingForm();
    initBindingListEvents();
    initReaderEvents();
    initRefreshButton();

    // 2. Connect with bridge if available
    try {
      await api.ready();
    } catch (e) {
      console.warn("[DocMemory] api.ready fallback:", e);
    }

    // 3. Load backend data
    try {
      await Promise.all([loadDocs(), loadBindings(), searchGroups("")]);
      console.log("[DocMemory] Initial data loaded successfully.");
    } catch (e) {
      console.warn("[DocMemory] Initial data load partial failure:", e);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startApp);
  } else {
    startApp();
  }
})();
