export interface BoundingBox {
  x_min: number | null;
  y_min: number | null;
  x_max: number | null;
  y_max: number | null;
}

export interface ClassificationResult {
  predicted_class: string;
  confidence: number;
  is_uncertain: boolean;
  probabilities: Record<string, number>;
}

export interface SegmentationResult {
  has_nodule: boolean;
  mask_base64: string;
  bounding_box: BoundingBox;
  detection_boxes: BoundingBox[];
  nodule_area_pixels: number;
  confidence_map: string;
}

export interface GradCAMResult {
  heatmap_base64: string;
  overlay_base64: string;
  explanation: string;
}

export interface PredictResponse {
  classification: ClassificationResult;
  segmentation: SegmentationResult;
  gradcam: GradCAMResult;
  processing_time_seconds: number;
  model_version: string;
  device_used: string;
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  gpu_available: boolean;
}

export type Stage = "idle" | "loading" | "results";

export interface AppState {
  file: File | null;
  result: PredictResponse | null;
  loading: boolean;
  error: string | null;
  uploadProgress: number;
  stage: Stage;
  setFile: (file: File | null) => void;
  setResult: (result: PredictResponse | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setUploadProgress: (progress: number) => void;
  setStage: (stage: Stage) => void;
  reset: () => void;
}
