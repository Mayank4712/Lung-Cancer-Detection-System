"use client";

import { useEffect, useMemo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, Wand2, XCircle } from "lucide-react";
import FileUploader from "./components/FileUploader";
import ImagePreview from "./components/ImagePreview";
import ClassificationResultDisplay from "./components/ClassificationResultDisplay";
import SegmentationViewer from "./components/SegmentationViewer";
import GradCAMViewer from "./components/GradCAMViewer";
import { predict } from "@/lib/api";
import { useStore } from "@/lib/store";

export default function Home() {
  const { file, result, loading, error, uploadProgress, stage, setError } =
    useStore();

  const originalSrc = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  const handlePredict = async () => {
    if (!file) return;
    setError(null);
    try {
      await predict(file);
    } catch (err) {
      const msg =
        err instanceof Error && err.message.includes("timeout")
          ? "Request timed out after 30 seconds. Please try again."
          : `Prediction failed: ${
              err instanceof Error ? err.message : "Unknown error"
            }`;
      setError(msg);
    }
  };

  return (
    <main className="min-h-screen px-4 py-10 sm:px-8">
      <div className="mx-auto max-w-6xl">
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-slate-100">
            Lung Cancer Detection
          </h1>
          <p className="mt-2 text-slate-400">
            Classification · Segmentation · Explainability (GradCAM)
          </p>
        </header>

        <AnimatePresence mode="wait">
          {stage === "loading" ? (
            <motion.section
              key="loading"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="flex flex-col items-center justify-center rounded-3xl border border-slate-800 bg-slate-900/40 p-16 text-center"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
              >
                <Activity className="h-10 w-10 text-accent-400" />
              </motion.div>
              <p className="mt-4 text-lg text-slate-200">
                Analyzing scan…
              </p>
              {uploadProgress > 0 && uploadProgress < 100 && (
                <p className="mt-1 text-sm text-slate-500">
                  Uploading {uploadProgress}%
                </p>
              )}
              <div className="mt-4 h-1.5 w-56 overflow-hidden rounded-full bg-slate-800">
                <motion.div
                  className="h-full bg-accent-500"
                  animate={{ width: ["10%", "90%"] }}
                  transition={{ repeat: Infinity, duration: 1.1, ease: "easeInOut" }}
                />
              </div>
            </motion.section>
          ) : stage === "results" && result ? (
            <motion.section
              key="results"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="flex flex-col gap-8"
            >
              <div className="flex flex-wrap items-center justify-between gap-4">
                <button
                  type="button"
                  onClick={() => useStore.getState().reset()}
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-accent-500/60"
                >
                  <Wand2 className="h-4 w-4" /> New scan
                </button>
                <div className="flex items-center gap-4 text-xs text-slate-500">
                  <span>
                    {result.processing_time_seconds.toFixed(3)}s
                  </span>
                  <span>{result.model_version}</span>
                  <span className="rounded-full bg-slate-800 px-2 py-0.5">
                    {result.device_used}
                  </span>
                </div>
              </div>

              <div className="grid gap-8 lg:grid-cols-2">
                <div className="space-y-8">
                  <ImagePreview />
                  <ClassificationResultDisplay result={result.classification} />
                </div>
                <div className="space-y-8">
                  <SegmentationViewer
                    result={result.segmentation}
                    originalSrc={originalSrc}
                  />
                  <GradCAMViewer
                    result={result.gradcam}
                    originalSrc={originalSrc}
                  />
                </div>
              </div>
            </motion.section>
          ) : (
            <motion.section
              key="upload"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="mx-auto max-w-2xl"
            >
              {error && (
                <div className="mb-4 flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
                  <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  {error}
                </div>
              )}

              {file ? (
                <div className="space-y-5">
                  <ImagePreview />
                  <button
                    type="button"
                    onClick={handlePredict}
                    disabled={loading}
                    className="w-full rounded-xl bg-accent-500 py-3 text-sm font-semibold text-slate-900 transition-colors hover:bg-accent-400 disabled:opacity-60"
                  >
                    {loading ? "Analyzing…" : "Run analysis"}
                  </button>
                </div>
              ) : (
                <FileUploader />
              )}
            </motion.section>
          )}
        </AnimatePresence>

        <div className="mt-12 text-center text-xs text-slate-600">
          This is a research/demo tool and is not a medical diagnostic device.
        </div>
      </div>
    </main>
  );
}
