"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
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
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="text-slate-300">{label}</span>
        <span className="tabular-nums text-slate-400">
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
        <motion.div
          className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.round(value * 100)}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
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

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-slate-700 bg-slate-900/60 p-5"
    >
      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
        Classification
      </h3>

      <div className="mb-4 flex items-center gap-3">
        {!result.is_uncertain ? (
          <CheckCircle2 className="h-5 w-5 text-emerald-400" />
        ) : (
          <AlertTriangle className="h-5 w-5 text-amber-400" />
        )}
        <span className="text-xl font-medium text-slate-100">
          {result.predicted_class}
        </span>
        {result.is_uncertain && (
          <span className="rounded-full bg-amber-500/15 px-3 py-1 text-xs font-medium text-amber-400">
            Uncertain
          </span>
        )}
      </div>

      <div className="space-y-4">
        <ProbabilityBar
          label="Healthy"
          value={healthy}
          color="bg-emerald-500"
        />
        <ProbabilityBar
          label="Lung Nodule"
          value={nodule}
          color="bg-accent-500"
        />
      </div>

      <p className="mt-4 text-xs text-slate-500">
        Confidence{" "}
        <span className="tabular-nums text-slate-300">
          {(result.confidence * 100).toFixed(1)}%
        </span>
      </p>
    </motion.div>
  );
}
