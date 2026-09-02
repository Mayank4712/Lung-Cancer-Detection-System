"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud, FileImage, X } from "lucide-react";
import { useStore } from "@/lib/store";

const ALLOWED_TYPES = ["image/png", "image/jpeg"];
const MAX_SIZE_MB = 10;

export default function FileUploader() {
  const inputRef = useRef<HTMLInputElement>(null);
  const { file, setFile, setError } = useStore();
  const [dragging, setDragging] = useState(false);

  const validate = useCallback(
    (f: File): string | null => {
      if (!ALLOWED_TYPES.includes(f.type)) {
        return "Unsupported file type. Please upload a PNG or JPG image.";
      }
      if (f.size > MAX_SIZE_MB * 1024 * 1024) {
        return `File exceeds ${MAX_SIZE_MB}MB limit.`;
      }
      return null;
    },
    []
  );

  const handleFile = useCallback(
    (f: File | undefined) => {
      if (!f) return;
      const err = validate(f);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      setFile(f);
    },
    [validate, setFile, setError]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      handleFile(e.dataTransfer.files?.[0]);
    },
    [handleFile]
  );

  return (
    <motion.div
      layout
      className="w-full"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg,image/png,image/jpeg"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {!file ? (
        <motion.button
          type="button"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          onClick={() => inputRef.current?.click()}
          className={`w-full rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-300 ${
            dragging
              ? "border-accent-400 bg-accent-500/10 shadow-glow"
              : "border-slate-700/80 bg-slate-900/40 backdrop-blur-md hover:border-accent-500/60 hover:bg-slate-900/70 hover:shadow-glow"
          }`}
        >
          <UploadCloud className="mx-auto mb-3 h-10 w-10 text-accent-400" />
          <p className="text-lg font-medium text-slate-200">
            Drag &amp; drop a CT scan, or click to browse
          </p>
          <p className="mt-1 text-sm text-slate-500">
            PNG or JPG · up to {MAX_SIZE_MB}MB
          </p>
        </motion.button>
      ) : (
        <motion.div
          layout
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-4 rounded-2xl border border-slate-800/80 bg-slate-900/60 p-4 backdrop-blur-md"
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-accent-500/15 ring-1 ring-accent-400/20">
            <FileImage className="h-6 w-6 text-accent-400" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-slate-200">
              {file.name}
            </p>
            <p className="text-xs text-slate-500">
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
          <button
            type="button"
            onClick={() => setFile(null)}
            className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
            aria-label="Remove file"
          >
            <X className="h-5 w-5" />
          </button>
        </motion.div>
      )}
    </motion.div>
  );
}
