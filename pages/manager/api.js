// xbdoc WebUI 网络层：桥接探测 + 后端 API + 上传/下载
// 被 app.js 依赖（index.html 里先加载本文件），通过 window.DocMemoryApi 暴露。
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

  window.DocMemoryApi = api;
})();
