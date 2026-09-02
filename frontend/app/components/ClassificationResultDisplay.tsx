"use client";

import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import type { ClassificationResult } from "@/lib/types";

interface Props {
  result: ClassificationResult;
}

function ProbabilityBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-sm">
        <span className="text-slate-300">{label}</span>
        <span className="tabular-nums text-slate-400">
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800/80 ring-1 ring-inset ring-slate-700/40">
        <motion.div
          className={`relative h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.round(value * 100)}%` }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        >
          {/* subtle top highlight for depth */}
          <span className="absolute inset-x-0 top-0 h-1/2 rounded-full bg-white/10" />
        </motion.div>
      </div>
    </div>
  );
}

export default function ClassificationResultDisplay({
  result,
}: Props) {
  const probs = result.probabilities;
  const healthy = probs["Healthy"] ?? 0;
  const nodule = probs["Lung_Nodule"] ?? 0;
  const isNodule = result.predicted_class === "Lung_Nodule";

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card glass-card-glow p-5"
    >
      <h3 className="mb-4 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
        Classification
      </h3>

      <div className="mb-5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {isNodule ? (
            <div className="rounded-xl bg-amber-500/15 p-2.5 ring-1 ring-amber-500/30">
              <ShieldAlert className="h-6 w-6 text-amber-400" />
            </div>
          ) : (
            <div className="rounded-xl bg-emerald-500/15 p-2.5 ring-1 ring-emerald-500/30">
              <ShieldCheck className="h-6 w-6 text-emerald-400" />
            </div>
          )}
          <div>
            <span className="text-xl font-medium text-slate-100">
              {result.predicted_class}
            </span>
            <span className="ml-2 text-[11px] uppercase tracking-wide text-slate-500">
              {isNodule ? "Nodule detected" : "No nodule"}
            </span>
          </div>
        </div>

        {isNodule ? (
          <span className="rounded-full bg-gradient-to-r from-amber-500/20 to-crimson/10 px-3 py-1 text-xs font-semibold text-amber-400 ring-1 ring-amber-500/30 shadow-glow">
            Nodule
          </span>
        ) : (
          <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-300 ring-1 ring-emerald-500/25">
            Healthy
          </span>
        )}
      </div>

      <div className="space-y-4">
        <ProbabilityBar
          label="Healthy"
          value={healthy}
          color="bg-gradient-to-r from-emerald-500 to-emerald-400"
        />
        <ProbabilityBar
          label="Lung Nodule"
          value={nodule}
          color="bg-gradient-to-r from-accent-600 to-accent-400"
        />
      </div>

      <p className="mt-4 flex items-center gap-1.5 text-xs text-slate-500">
        {result.is_uncertain ? (
          <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
        ) : (
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        )}
        Confidence{" "}
        <span className="tabular-nums text-slate-300">
          {(result.confidence * 100).toFixed(1)}%
        </span>
      </p>
    </motion.div>
  );
}
