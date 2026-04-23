/**
 * Supported YOLO model weights.
 * Ultralytics auto-downloads them on first use.
 *
 * Grouped for <optgroup> rendering; ordered newest → oldest within each family.
 */
export const YOLO_MODEL_GROUPS = [
    {
        family: "YOLO26",
        note: "2025 — edge-optimised, NMS-free",
        models: [
            { value: "yolo26n.pt", label: "YOLO26 Nano — fastest edge",  params: "~3M"   },
            { value: "yolo26s.pt", label: "YOLO26 Small",                params: "~10M"  },
            { value: "yolo26m.pt", label: "YOLO26 Medium",               params: "~20M"  },
            { value: "yolo26l.pt", label: "YOLO26 Large",                params: "~28M"  },
            { value: "yolo26x.pt", label: "YOLO26 XL — best accuracy",  params: "~57M"  },
        ],
    },
    {
        family: "YOLO12",
        note: "NeurIPS 2025 — attention-centric",
        models: [
            { value: "yolo12n.pt", label: "YOLO12 Nano",                 params: "2.6M"  },
            { value: "yolo12s.pt", label: "YOLO12 Small",                params: "9.3M"  },
            { value: "yolo12m.pt", label: "YOLO12 Medium",               params: "20.2M" },
            { value: "yolo12l.pt", label: "YOLO12 Large",                params: "26.4M" },
            { value: "yolo12x.pt", label: "YOLO12 XL",                   params: "59.1M" },
        ],
    },
    {
        family: "YOLO11",
        note: "Stable — recommended",
        models: [
            { value: "yolo11n.pt", label: "YOLO11 Nano — fastest",      params: "2.6M"  },
            { value: "yolo11s.pt", label: "YOLO11 Small",                params: "9.4M"  },
            { value: "yolo11m.pt", label: "YOLO11 Medium",               params: "20.1M" },
            { value: "yolo11l.pt", label: "YOLO11 Large",                params: "25.3M" },
            { value: "yolo11x.pt", label: "YOLO11 XL — best accuracy",  params: "56.9M" },
        ],
    },
    {
        family: "YOLOv10",
        models: [
            { value: "yolov10n.pt", label: "YOLOv10 Nano",   params: "2.3M"  },
            { value: "yolov10s.pt", label: "YOLOv10 Small",  params: "7.2M"  },
            { value: "yolov10m.pt", label: "YOLOv10 Medium", params: "15.4M" },
            { value: "yolov10b.pt", label: "YOLOv10 Base",   params: "19.1M" },
            { value: "yolov10l.pt", label: "YOLOv10 Large",  params: "24.4M" },
            { value: "yolov10x.pt", label: "YOLOv10 XL",     params: "29.5M" },
        ],
    },
    {
        family: "YOLOv9",
        models: [
            { value: "yolov9c.pt", label: "YOLOv9 C",                 params: "25.3M" },
            { value: "yolov9e.pt", label: "YOLOv9 E — high accuracy", params: "57.3M" },
        ],
    },
    {
        family: "YOLOv8",
        models: [
            { value: "yolov8n.pt", label: "YOLOv8 Nano",   params: "3.2M"  },
            { value: "yolov8s.pt", label: "YOLOv8 Small",  params: "11.2M" },
            { value: "yolov8m.pt", label: "YOLOv8 Medium", params: "25.9M" },
            { value: "yolov8l.pt", label: "YOLOv8 Large",  params: "43.7M" },
            { value: "yolov8x.pt", label: "YOLOv8 XL",     params: "68.2M" },
        ],
    },
];

/** Flat list of all models (same data, useful for lookup) */
export const YOLO_MODELS_FLAT = YOLO_MODEL_GROUPS.flatMap(g =>
    g.models.map(m => ({ ...m, family: g.family }))
);

/** Default model for seed training.
 *  Changed from yolo11n → yolo11s: the nano model lacks capacity to learn
 *  subtle inspection cues (e.g. white-clip visibility on water pipes).
 *  Small adds ~3× more parameters for a modest speed trade-off. */
export const DEFAULT_SEED_MODEL = "yolo11s.pt";

/** Default model for main training (larger for best accuracy) */
export const DEFAULT_MAIN_MODEL = "yolo11s.pt";
