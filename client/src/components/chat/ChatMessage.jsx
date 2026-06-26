import styles from "./ChatMessage.module.css";

export default function ChatMessage({ message }) {
  const { role, content, files = [], sources } = message;

  const containerClass =
    role === "user" ? styles.userContainer : styles.botContainer;
  const messageClass =
    role === "user" ? styles.userMessage : styles.botMessage;

  return (
    <div className={`${styles.messageWrapper} ${containerClass}`}>
      {content && <div className={messageClass}>{content}</div>}

      {files.length > 0 && (
        <div className={styles.filesWrapper}>
          {files.map((f, idx) => (
            <div key={idx} className={styles.fileItem}>
              📎 <span
                className={f.uploaded ? styles.uploadedFile : styles.uploadingFile}
              >
                {f.file.name}
              </span>
              <span className={styles.fileSize}>
                ({Math.round(f.file.size / 1024)} KB)
              </span>

              {!f.uploaded && !f.error && (
                <div className={styles.progressBarWrapper}>
                  <div
                    className={styles.progressBar}
                    style={{ width: `${f.progress}%` }}
                  />
                </div>
              )}

              {f.error && (
                <span className={styles.errorMessage}>❌ {f.error}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {sources && sources.length > 0 && (
        <ul className={styles.sourcesList}>
          {sources.map((s, idx) => (
            <li key={idx}>
              {s.chunk.slice(0, 120)}
              {s.chunk.length > 120 ? "..." : ""}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
