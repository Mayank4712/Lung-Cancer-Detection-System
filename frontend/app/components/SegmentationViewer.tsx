"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Layers, Eye, ScanLine, ChevronRight } from "lucide-react";
import type { BoundingBox, SegmentationResult } from "@/lib/types";

interface Props {
  result: SegmentationResult;
  originalSrc: string;
  predictedClass: string;
}

export default function SegmentationViewer({
  result,
  originalSrc,
  predictedClass,
}: Props) {
  const [showMask, setShowMask] = useState(true);
  const [showBox, setShowBox] = useState(true);
  const [opacity, setOpacity] = useState(0.5);

  const boxes = useMemo(() => result.detection_boxes ?? [], [result.detection_boxes]);
  const showDetections =
    predictedClass === "Lung_Nodule" && boxes.length > 0;

  const boxStyles = useMemo(() => {
    if (!showDetections) return [];
    return boxes
      .filter(
        (b: BoundingBox) =>
          b.x_min !== null &&
          b.y_min !== null &&
          b.x_max !== null &&
          b.y_max !== null,
      )
      .map((b: BoundingBox) => ({
        left: `${b.x_min!}px`,
        top: `${b.y_min!}px`,
        width: `${b.x_max! - b.x_min!}px`,
        height: `${b.y_max! - b.y_min!}px`,
      }));
  }, [boxes, showDetections]);

  const fill = `${Math.round(opacity * 100)}%`;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card glass-card-glow p-5"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
          <ScanLine className="h-4 w-4 text-accent-400" />
          Segmentation
        </h3>
        <span className="rounded-full bg-slate-800/80 px-2.5 py-1 text-xs tabular-nums text-slate-300 ring-1 ring-slate-700/50">
          {result.nodule_area_pixels.toLocaleString()} px²
        </span>
      </div>

      <div className="viewport-frame relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
        <img
          src={originalSrc}
          alt="Original"
          className="w-full object-contain"
        />
        {showMask && (
          <img
            src={result.mask_base64}
            alt="Segmentation mask"
            className="absolute inset-0 h-full w-full object-contain"
            style={{ opacity }}
          />
        )}
        {showBox &&
          showDetections &&
          boxStyles.map((style, i) => (
            <div
              key={i}
              className="absolute border-2 border-accent-400 shadow-glow"
              style={{ ...style, pointerEvents: "none" }}
            />
          ))}
        {!showDetections && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/50">
            <p className="text-sm text-slate-300">No nodule detected</p>
          </div>
        )}
      </div>

      <div className="mt-5 space-y-4">
        {/* Mask opacity slider */}
        <div>
          <div className="mb-2 flex items-center justify-between text-sm">
            <label
              className="flex items-center gap-2 text-slate-300"
              htmlFor="seg-opacity"
            >
              <Layers className="h-4 w-4 text-cyan-400" /> Mask overlay
            </label>
            <span className="tabular-nums text-slate-400">
              {Math.round(opacity * 100)}%
            </span>
          </div>
          <input
            id="seg-opacity"
            type="range"
            min={0}
            max={100}
            value={Math.round(opacity * 100)}
            onChange={(e) => setOpacity(Number(e.target.value) / 100)}
            className="ui-slider"
            style={{ ["--fill" as string]: fill }}
            aria-label="Mask opacity"
          />
        </div>

        {/* Toggle pills */}
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => setShowMask((v) => !v)}
            className={`inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm transition-all ${
              showMask
                ? "bg-accent-500/15 text-accent-300 ring-1 ring-accent-500/30"
                : "bg-slate-800/60 text-slate-400 hover:text-slate-200"
            }`}
          >
            <Layers className="h-3.5 w-3.5" /> Mask
          </button>
          <button
            type="button"
            onClick={() => setShowBox((v) => !v)}
            className={`inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm transition-all ${
              showBox
                ? "bg-accent-500/15 text-accent-300 ring-1 ring-accent-500/30"
                : "bg-slate-800/60 text-slate-400 hover:text-slate-200"
            }`}
          >
            <Eye className="h-3.5 w-3.5" /> Bounding box
          </button>
        </div>

        {showDetections && (
          <p className="flex items-center gap-1.5 text-xs text-slate-400">
            <ChevronRight className="h-3 w-3 text-accent-400" />
            {boxes.length} region{boxes.length > 1 ? "s" : ""} located
          </p>
        )}
      </div>
    </motion.div>
  );
}
