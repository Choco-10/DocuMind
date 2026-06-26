import { useEffect, useMemo, useState } from "react";
import ChatWindow from "../components/chat/ChatWindow";
import ChatInput from "../components/chat/ChatInput";
import useChatStore from "../store/chatStore";
import {
  clearDocuments,
  deleteDocument,
  getDocuments,
  streamRagQuery,
  uploadDocument,
} from "../api/ragApi";
import styles from "./Chat.module.css";

export default function Chat() {
  const addMessage = useChatStore((state) => state.addMessage);
  const updateFileProgress = useChatStore((state) => state.updateFileProgress);
  const markFileUploaded = useChatStore((state) => state.markFileUploaded);
  const markFileError = useChatStore((state) => state.markFileError);
  const updateLastAssistantMessage = useChatStore(
    (state) => state.updateLastAssistantMessage
  );
  const resetChat = useChatStore((state) => state.reset);

  const [loading, setLoading] = useState(false);
  const [docsLoading, setDocsLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [docError, setDocError] = useState("");
  const [busyDocKey, setBusyDocKey] = useState("");
  const [sessionId, setSessionId] = useState(() =>
    crypto?.randomUUID ? crypto.randomUUID() : String(Date.now())
  );

  const handleNewChat = () => {
    resetChat();
    setSessionId(crypto?.randomUUID ? crypto.randomUUID() : String(Date.now()));
  };

  const totalChunks = useMemo(
    () => documents.reduce((sum, doc) => sum + (doc.chunks || 0), 0),
    [documents]
  );

  const loadDocuments = async () => {
    setDocsLoading(true);
    setDocError("");
    try {
      const data = await getDocuments();
      setDocuments(Array.isArray(data?.documents) ? data.documents : []);
    } catch (err) {
      setDocError(err?.response?.data?.detail || "Failed to load documents");
    } finally {
      setDocsLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleDeleteDoc = async (source) => {
    setBusyDocKey(source);
    setDocError("");
    try {
      await deleteDocument(source);
      setDocuments((prev) => prev.filter((doc) => doc.source !== source));
    } catch (err) {
      setDocError(err?.response?.data?.detail || "Failed to delete document");
    } finally {
      setBusyDocKey("");
    }
  };

  const handleClearDocs = async () => {
    setBusyDocKey("ALL");
    setDocError("");
    try {
      await clearDocuments();
      setDocuments([]);
    } catch (err) {
      setDocError(err?.response?.data?.detail || "Failed to clear documents");
    } finally {
      setBusyDocKey("");
    }
  };

  const handleSend = async ({ content, files }) => {
    if (!content && (!files || files.length === 0)) return;

    const messageId = addMessage({
      role: "user",
      content: content || "",
      files: files || [],
    });

    if (files && files.length > 0) {
      for (let idx = 0; idx < files.length; idx++) {
        const file = files[idx];
        try {
          await uploadDocument(file, (percent) => {
            updateFileProgress(messageId, idx, percent);
          });
          markFileUploaded(messageId, idx);
        } catch (err) {
          markFileError(messageId, idx, err?.message || "Upload failed");
        }
      }

      await loadDocuments();
    }

    if (content) {
      addMessage({ role: "assistant", content: "" });
      setLoading(true);
      try {
        await streamRagQuery({
          question: content,
          session_id: sessionId,
          top_k: 5,
          onToken: (token) => {
            updateLastAssistantMessage(token);
          },
          onMeta: (meta) => {
            if (meta.sources) {
              console.log("Sources:", meta.sources);
            }
          },
        });
      } catch (err) {
        console.error("Chat query failed:", err);
        updateLastAssistantMessage("Error generating answer.");
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className={styles.container}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <h3 className={styles.sidebarTitle}>Context Documents</h3>
          <div className={styles.sidebarActions}>
            <button
              className={styles.sidebarButton}
              onClick={loadDocuments}
              disabled={docsLoading || !!busyDocKey}
            >
              Refresh
            </button>
            <button
              className={styles.sidebarDangerButton}
              onClick={handleClearDocs}
              disabled={!documents.length || !!busyDocKey}
            >
              {busyDocKey === "ALL" ? "Clearing..." : "Clear"}
            </button>
          </div>
        </div>

        <p className={styles.contextHint}>
          Used automatically for every chat response.
        </p>

        <p className={styles.sidebarStats}>
          Files: <strong>{documents.length}</strong> | Chunks: <strong>{totalChunks}</strong>
        </p>

        {docError && <p className={styles.docError}>{docError}</p>}

        <div className={styles.docsList}>
          {docsLoading ? (
            <div className={styles.docsEmpty}>Loading...</div>
          ) : documents.length === 0 ? (
            <div className={styles.docsEmpty}>No documents indexed.</div>
          ) : (
            documents.map((doc) => (
              <div className={styles.docItem} key={doc.source}>
                <div className={styles.docMeta}>
                  <div className={styles.docName} title={doc.source}>{doc.source}</div>
                  <div className={styles.docChunks}>{doc.chunks} chunks</div>
                </div>
                <button
                  className={styles.docRemoveButton}
                  onClick={() => handleDeleteDoc(doc.source)}
                  disabled={!!busyDocKey}
                >
                  {busyDocKey === doc.source ? "..." : "Remove"}
                </button>
              </div>
            ))
          )}
        </div>
      </aside>

      <section className={styles.chatPanel}>
        <div className={styles.header}>
          <h2 className={styles.title}>RAG Chat</h2>
          <button className={styles.newChatButton} onClick={handleNewChat}>
            New Chat
          </button>
        </div>

        <div className={styles.chatWindow}>
          <ChatWindow loading={loading} />
        </div>

        <div className={styles.chatInputWrapper}>
          <ChatInput onSend={handleSend} />
        </div>
      </section>
    </div>
  );
}