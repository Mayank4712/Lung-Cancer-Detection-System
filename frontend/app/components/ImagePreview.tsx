"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ZoomIn, ZoomOut, Move, RotateCcw, Crosshair } from "lucide-react";
import { useStore } from "@/lib/store";

export default function ImagePreview() {
  const file = useStore((s) => s.file);
  const [src, setSrc] = useState<string | null>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [start, setStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!file) {
      setSrc(null);
      setScale(1);
      setOffset({ x: 0, y: 0 });
      return;
    }
    const url = URL.createObjectURL(file);
    setSrc(url);
    setScale(1);
    setOffset({ x: 0, y: 0 });
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const clampScale = (s: number) => Math.min(4, Math.max(1, s));

  const onWheel = (e: React.WheelEvent) => {
    const next = clampScale(scale - e.deltaY * 0.001);
    setScale(next);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    setDragging(true);
    setStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging) return;
    setOffset({ x: e.clientX - start.x, y: e.clientY - start.y });
  };

  const onPointerUp = () => setDragging(false);

  const viewportStyle = useMemo(() => {
    const max = (scale - 1) * 160;
    const x = Math.max(-max, Math.min(max, offset.x));
    const y = Math.max(-max, Math.min(max, offset.y));
    return {
      transform: `scale(${scale}) translate(${x}px, ${y}px)`,
      cursor: scale > 1 ? "grab" : "default",
    };
  }, [scale, offset]);

  if (!src) return null;

  return (
    <motion.div
      layout
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass-card glass-card-glow overflow-hidden"
    >
      <div className="flex items-center justify-between border-b border-slate-800/70 px-3 py-2">
        <span className="flex items-center gap-2 text-xs text-slate-400">
          <Crosshair className="h-3.5 w-3.5 text-accent-400" /> Drag to pan ·
          scroll to zoom
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setScale(clampScale(scale - 0.25))}
            className="rounded-md p-1.5 text-slate-300 transition-colors hover:bg-slate-800/80 hover:text-accent-300"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="w-10 text-center text-xs tabular-nums text-slate-400">
            {Math.round(scale * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setScale(clampScale(scale + 0.25))}
            className="rounded-md p-1.5 text-slate-300 transition-colors hover:bg-slate-800/80 hover:text-accent-300"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <span className="mx-1 h-4 w-px bg-slate-700/70" />
          <button
            type="button"
            onClick={() => {
              setScale(1);
              setOffset({ x: 0, y: 0 });
            }}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-accent-400 transition-colors hover:bg-slate-800/80"
          >
            <RotateCcw className="h-3.5 w-3.5" /> Reset
          </button>
        </div>
      </div>

      <div className="relative flex items-center justify-center bg-slate-950">
        <div
          className="viewport-frame relative flex h-72 w-full items-center justify-center overflow-hidden"
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
          style={{ touchAction: "none" }}
        >
          <Move
            className="pointer-events-none absolute bottom-2 right-2 z-10 h-4 w-4 text-slate-600"
          />
          <img
            src={src}
            alt="CT scan preview"
            className="max-h-full max-w-full select-none object-contain"
            style={viewportStyle}
            draggable={false}
          />
        </div>
      </div>
    </motion.div>
  );
}
