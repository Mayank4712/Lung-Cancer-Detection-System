import axios from "axios";
import { useStore } from "./store";
import type { HealthResponse, PredictResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const http = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30-second timeout
});

const isAborted = (error: { code?: string }) =>
  axios.isCancel(error) || error?.code === "ECONNABORTED";

export async function predict(image: File): Promise<PredictResponse> {
  useStore.setState({
    loading: true,
    stage: "loading",
    error: null,
    result: null,
    uploadProgress: 0,
  });

  const formData = new FormData();
  formData.append("image", image);

  const res = await http.post<PredictResponse>("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (event) => {
      if (event.total) {
        useStore.setState({
          uploadProgress: Math.round((event.loaded / event.total) * 100),
        });
      }
    },
  });

  useStore.setState({
    result: res.data,
    loading: false,
    stage: "results",
  });
  return res.data;
}

export async function healthCheck(): Promise<HealthResponse> {
  const res = await http.get<HealthResponse>("/health");
  return res.data;
}

export { isAborted };
