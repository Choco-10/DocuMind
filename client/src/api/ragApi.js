import api from "./axios";

export const uploadDocument = async (file, onProgress) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await api.post("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (event) => {
      if (event.total) {
        const percent = Math.round((event.loaded * 100) / event.total);
        onProgress?.(percent);
      }
    },
  });

  return res.data;
};

export const getDocuments = async () => {
  const res = await api.get("/api/upload/documents");
  return res.data;
};

export const deleteDocument = async (source) => {
  const res = await api.delete(`/api/upload/documents/${encodeURIComponent(source)}`);
  return res.data;
};

export const clearDocuments = async () => {
  const res = await api.delete("/api/upload/documents");
  return res.data;
};

export const streamRagQuery = async ({ question, session_id, top_k = 5, onToken, onMeta }) => {
  const baseURL = api.defaults.baseURL || "http://localhost:8000";
  const res = await fetch(`${baseURL}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id, top_k }),
  });

  if (!res.ok || !res.body) {
    throw new Error("Streaming request failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    try {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const event of events) {
        const line = event.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;

        const payload = line.replace("data: ", "").trim();
        if (payload === "[DONE]") return;

        try {
          const parsed = JSON.parse(payload);
          if (parsed.token) onToken?.(parsed.token);
          if (parsed.sources || parsed.done) onMeta?.(parsed);
        } catch (parseErr) {
          console.error("Failed to parse streaming payload:", parseErr);
        }
      }
    } catch (err) {
      console.error("Stream reading error:", err);
      throw err;
    }
  }
};