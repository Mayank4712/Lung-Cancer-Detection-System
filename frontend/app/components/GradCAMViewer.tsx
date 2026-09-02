"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Flame, Info, Layers2 } from "lucide-react";
import type { GradCAMResult } from "@/lib/types";

type Mode = "heatmap" | "overlay";

interface Props {
  result: GradCAMResult;
  originalSrc: string;
}

const TABS: { id: Mode; label: string }[] = [
  { id: "heatmap", label: "Heatmap" },
  { id: "overlay", label: "Overlay" },
];

export default function GradCAMViewer({ result, originalSrc }: Props) {
  const [mode, setMode] = useState<Mode>("overlay");
  const [opacity, setOpacity] = useState(0.4);

  const activeIndex = TABS.findIndex((t) => t.id === mode);
  const fill = `${Math.round(opacity * 100)}%`;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card glass-card-glow p-5"
    >
      <div className="mb-4 flex items-center gap-2">
        <Flame className="h-4 w-4 text-accent-400" />
        <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
          GradCAM
        </h3>
      </div>

      <div className="viewport-frame relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
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

      <div className="mt-5 space-y-4">
        {/* Animated pill tab selector */}
        <div className="relative grid grid-cols-2 rounded-lg bg-slate-800/60 p-1 ring-1 ring-slate-700/40">
          {/* sliding background indicator */}
          <motion.div
            className="absolute inset-y-1 w-[calc(50%-4px)] rounded-md bg-slate-700/80 shadow-glow"
            animate={{ left: `${activeIndex * 50}%` }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
          />
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setMode(tab.id)}
              className={`relative z-10 flex items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm capitalize transition-colors ${
                mode === tab.id
                  ? "font-medium text-slate-100"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.id === "heatmap" ? (
                <Flame className="h-3.5 w-3.5" />
              ) : (
                <Layers2 className="h-3.5 w-3.5" />
              )}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Transparency slider (overlay only) */}
        {mode === "overlay" && (
          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-slate-300">Transparency</span>
              <span className="tabular-nums text-slate-400">
                {Math.round(opacity * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(opacity * 100)}
              onChange={(e) => setOpacity(Number(e.target.value) / 100)}
              className="ui-slider"
              style={{ ["--fill" as string]: fill }}
              aria-label="Heatmap transparency"
            />
          </div>
        )}
      </div>

      <p className="mt-4 flex items-start gap-2 text-xs leading-relaxed text-slate-500">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent-400" />
        {result.explanation}
      </p>
    </motion.div>
  );
}
