import { useState } from "react";
import styles from "./ChatInput.module.css";

export default function ChatInput({ onSend }) {
  const [value, setValue] = useState("");
  const [files, setFiles] = useState([]);
  const [isSending, setIsSending] = useState(false);

  const handleFileChange = (e) => {
    const newFiles = Array.from(e.target.files);
    setFiles((prev) => [...prev, ...newFiles]);
    e.target.value = "";
  };

  const removeFile = (idx) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleSend = async () => {
    if (!value.trim() && files.length === 0) return;
    if (isSending) return;

    const contentToSend = value.trim();
    const filesToSend = files;

    setValue("");
    setFiles([]);

    setIsSending(true);
    try {
      await onSend({ content: contentToSend, files: filesToSend });
    } catch (err) {
      console.error("Send failed:", err);

      setValue(contentToSend);
      setFiles(filesToSend);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className={styles.container}>
      <textarea
        className={styles.textarea}
        rows={1}
        placeholder="Type a message or attach files..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />

      <input
        type="file"
        multiple
        accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff,.tif,.webp"
        onChange={handleFileChange}
        className={styles.fileInput}
      />

      {files.length > 0 && (
        <div className={styles.filePreview}>
          {files.map((file, idx) => (
            <div key={idx} className={styles.fileItem}>
              <span>{file.name}</span>
              <button onClick={() => removeFile(idx)}>✕</button>
            </div>
          ))}
        </div>
      )}

      <button className={styles.button} onClick={handleSend} disabled={isSending}>
        {isSending ? "Sending..." : "Send"}
      </button>
    </div>
  );
}