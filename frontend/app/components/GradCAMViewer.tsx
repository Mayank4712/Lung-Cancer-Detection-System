"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Flame, Info } from "lucide-react";
import type { GradCAMResult } from "@/lib/types";

type Mode = "heatmap" | "overlay";

interface Props {
  result: GradCAMResult;
  originalSrc: string;
}

export default function GradCAMViewer({ result, originalSrc }: Props) {
  const [mode, setMode] = useState<Mode>("overlay");
  const [opacity, setOpacity] = useState(0.4);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-slate-700 bg-slate-900/60 p-5"
    >
      <div className="mb-4 flex items-center gap-2">
        <Flame className="h-4 w-4 text-accent-400" />
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          GradCAM
        </h3>
      </div>

      <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
        {mode === "overlay" ? (
          <>
            <img
              src={originalSrc}
              alt="Original"
              className="w-full object-contain"
            />
            <img
              src={result.heatmap_base64}
              alt="GradCAM heatmap"
              className="absolute inset-0 h-full w-full object-contain"
              style={{ opacity }}
            />
          </>
        ) : (
          <img
            src={result.heatmap_base64}
            alt="GradCAM heatmap"
            className="w-full object-contain"
          />
        )}
      </div>

      <div className="mt-4 space-y-3">
        <div className="flex gap-2">
          {(["heatmap", "overlay"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`flex-1 rounded-lg px-3 py-1.5 text-sm capitalize transition-colors ${
                mode === m
                  ? "bg-accent-500/20 text-accent-300"
                  : "bg-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {mode === "overlay" && (
          <div>
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="text-slate-300">Transparency</span>
              <span className="text-xs text-slate-500">
                {Math.round(opacity * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(opacity * 100)}
              onChange={(e) => setOpacity(Number(e.target.value) / 100)}
              className="w-full accent-cyan-500"
              aria-label="Heatmap transparency"
            />
          </div>
        )}
      </div>

      <p className="mt-4 flex items-start gap-2 text-xs leading-relaxed text-slate-500">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        {result.explanation}
      </p>
    </motion.div>
  );
}
