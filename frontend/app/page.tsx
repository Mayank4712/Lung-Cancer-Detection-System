"use client";

import { useEffect, useMemo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Cpu,
  Gauge,
  ShieldCheck,
  Tag,
  Wand2,
  XCircle,
  HeartPulse,
} from "lucide-react";
import FileUploader from "./components/FileUploader";
import ImagePreview from "./components/ImagePreview";
import ClassificationResultDisplay from "./components/ClassificationResultDisplay";
import SegmentationViewer from "./components/SegmentationViewer";
import GradCAMViewer from "./components/GradCAMViewer";
import { predict } from "@/lib/api";
import { useStore } from "@/lib/store";

const RESULT_STAGGER = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0 },
};

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

  const deviceLabel =
    result?.device_used === "cuda" ? "GPU" : result?.device_used ?? "CPU";

  return (
    <main className="relative min-h-screen overflow-x-hidden">
      {/* Ambient background glows */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute -top-40 left-1/2 h-[480px] w-[820px] -translate-x-1/2 rounded-full bg-accent-500/10 blur-[120px]" />
        <div className="absolute bottom-0 right-0 h-[360px] w-[520px] rounded-full bg-emerald-500/5 blur-[120px]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(19,25,39,0.6),transparent_60%)]" />
      </div>

      {/* Top navigation */}
      <header className="sticky top-0 z-30 border-b border-slate-800/60 bg-slate-950/60 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent-500/25 to-emerald-500/20 ring-1 ring-accent-400/30">
              <HeartPulse className="h-5 w-5 text-accent-400" />
            </div>
            <div className="leading-tight">
              <h1 className="text-sm font-semibold tracking-tight text-slate-100">
                Lung Cancer Detection
              </h1>
              <p className="text-[11px] text-slate-500">
                Diagnostic Vision Suite
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-2 sm:flex">
            {/* Device status pill */}
            <span className="flex items-center gap-1.5 rounded-full border border-slate-700/70 bg-slate-900/60 px-2.5 py-1 text-[11px] font-medium text-slate-300">
              <Cpu
                className={`h-3 w-3 ${
                  deviceLabel === "GPU"
                    ? "text-emerald-400"
                    : "text-cyan-400"
                }`}
              />
              {deviceLabel}
            </span>
            {/* Latency / status pill */}
            <span className="flex items-center gap-1.5 rounded-full border border-slate-700/70 bg-slate-900/60 px-2.5 py-1 text-[11px] font-medium text-slate-300">
              <Gauge className="h-3 w-3 text-slate-400" />
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
              </span>
              Online
            </span>
            {/* Version pill */}
            <span className="flex items-center gap-1.5 rounded-full border border-slate-700/70 bg-slate-900/60 px-2.5 py-1 text-[11px] font-medium text-slate-300">
              <Tag className="h-3 w-3 text-slate-400" />
              {result?.model_version ?? "v1.0.0"}
            </span>
            {/* Trusted / research pill */}
            <span className="flex items-center gap-1.5 rounded-full border border-slate-700/70 bg-emerald-500/5 px-2.5 py-1 text-[11px] font-medium text-emerald-300">
              <ShieldCheck className="h-3 w-3 text-emerald-400" />
              Research
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-8">
        <AnimatePresence mode="wait">
          {stage === "loading" ? (
            <motion.section
              key="loading"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="flex flex-col items-center justify-center rounded-3xl border border-slate-800 bg-slate-900/40 p-16 text-center backdrop-blur-md"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
              >
                <Activity className="h-10 w-10 text-accent-400" />
              </motion.div>
              <p className="mt-4 text-lg text-slate-200">Analyzing scan…</p>
              {uploadProgress > 0 && uploadProgress < 100 && (
                <p className="mt-1 text-sm text-slate-500">
                  Uploading {uploadProgress}%
                </p>
              )}
              <div className="mt-4 h-1.5 w-56 overflow-hidden rounded-full bg-slate-800">
                <motion.div
                  className="h-full bg-gradient-to-r from-accent-500 to-emerald-500"
                  animate={{ width: ["10%", "90%"] }}
                  transition={{ repeat: Infinity, duration: 1.1, ease: "easeInOut" }}
                />
              </div>
              {/* Skeleton shimmer over viewers while processing */}
              <div className="mt-8 grid w-full gap-6 lg:grid-cols-2">
                <div className="space-y-6">
                  <div className="skeleton h-72 rounded-2xl" />
                  <div className="skeleton h-40 rounded-2xl" />
                </div>
                <div className="space-y-6">
                  <div className="skeleton h-64 rounded-2xl" />
                  <div className="skeleton h-40 rounded-2xl" />
                </div>
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
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-accent-500/60 hover:bg-slate-800/60"
                >
                  <Wand2 className="h-4 w-4" /> New scan
                </button>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  <span className="tabular-nums">
                    {result.processing_time_seconds.toFixed(3)}s
                  </span>
                  <span className="rounded-full bg-slate-800/80 px-2.5 py-1">
                    {result.device_used}
                  </span>
                  <span className="rounded-full bg-slate-800/80 px-2.5 py-1">
                    {result.model_version}
                  </span>
                </div>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <motion.div
                  variants={RESULT_STAGGER}
                  initial="hidden"
                  animate="show"
                  transition={{ delay: 0.05 }}
                  className="space-y-6"
                >
                  <ImagePreview />
                  <ClassificationResultDisplay result={result.classification} />
                </motion.div>
                <motion.div
                  variants={RESULT_STAGGER}
                  initial="hidden"
                  animate="show"
                  transition={{ delay: 0.15 }}
                  className="space-y-6"
                >
                  <SegmentationViewer
                    result={result.segmentation}
                    originalSrc={originalSrc}
                    predictedClass={result.classification.predicted_class}
                  />
                  <GradCAMViewer
                    result={result.gradcam}
                    originalSrc={originalSrc}
                  />
                </motion.div>
              </div>
            </motion.section>
          ) : (
            <motion.section
              key="upload"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="mx-auto max-w-2xl pt-4"
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
                    className="w-full rounded-xl bg-gradient-to-r from-accent-500 to-emerald-500 py-3 text-sm font-semibold text-slate-950 transition-all hover:from-accent-400 hover:to-emerald-400 hover:shadow-glow disabled:opacity-60"
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
