"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Layers } from "lucide-react";
import type { SegmentationResult } from "@/lib/types";

interface Props {
  result: SegmentationResult;
  originalSrc: string;
}

export default function SegmentationViewer({ result, originalSrc }: Props) {
  const [showMask, setShowMask] = useState(true);
  const [showBox, setShowBox] = useState(true);
  const [opacity, setOpacity] = useState(0.5);

  const box = result.bounding_box;
  const hasBox =
    box.x_min !== null &&
    box.y_min !== null &&
    box.x_max !== null &&
    box.y_max !== null;

  const boxStyle = useMemo(() => {
    if (!hasBox) return undefined;
    const w = box.x_max! - box.x_min!;
    const h = box.y_max! - box.y_min!;
    return {
      left: `${box.x_min!}px`,
      top: `${box.y_min!}px`,
      width: `${w}px`,
      height: `${h}px`,
    };
  }, [box, hasBox]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-slate-700 bg-slate-900/60 p-5"
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Segmentation
        </h3>
        <span className="rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
          {result.nodule_area_pixels.toLocaleString()} px²
        </span>
      </div>

      <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
        <img src={originalSrc} alt="Original" className="w-full object-contain" />
        {showMask && (
          <img
            src={result.mask_base64}
            alt="Segmentation mask"
            className="absolute inset-0 h-full w-full object-contain"
            style={{ opacity }}
          />
        )}
        {showBox && hasBox && (
          <div
            className="absolute border-2 border-accent-400"
            style={{ ...boxStyle, pointerEvents: "none" }}
          />
        )}
        {!result.has_nodule && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/50">
            <p className="text-sm text-slate-300">No nodule detected</p>
          </div>
        )}
      </div>

      <div className="mt-4 space-y-3">
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={showMask}
              onChange={(e) => setShowMask(e.target.checked)}
              className="accent-cyan-500"
            />
            <Layers className="h-4 w-4" /> Mask overlay
          </label>
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
          aria-label="Mask opacity"
        />
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={showBox}
            onChange={(e) => setShowBox(e.target.checked)}
            className="accent-cyan-500"
          />
          Bounding box
        </label>
      </div>
    </motion.div>
  );
}
