import { create } from "zustand";
import type { AppState } from "./types";

export const useStore = create<AppState>((set) => ({
  file: null,
  result: null,
  loading: false,
  error: null,
  uploadProgress: 0,
  stage: "idle",
  setFile: (file) => set({ file }),
  setResult: (result) => set({ result }),
  setLoading: (loading) =>
    set({ loading, stage: loading ? "loading" : "idle" }),
  setError: (error) => set({ error }),
  setUploadProgress: (uploadProgress) => set({ uploadProgress }),
  setStage: (stage) => set({ stage }),
  reset: () =>
    set({
      file: null,
      result: null,
      loading: false,
      error: null,
      uploadProgress: 0,
      stage: "idle",
    }),
}));
