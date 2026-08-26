import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Stage, Layer, Rect, Text, Image, Group, Transformer, Line, Circle } from 'react-konva';
import useImage from 'use-image';
import TrainingPanel from './TrainingPanel';
import AutoAnnotatePanel from './AutoAnnotatePanel';
import MainTrainingPanel from './MainTrainingPanel';
import LabelsPanel from './LabelsPanel';
import ModelsPanel from './ModelsPanel';
import ReviewPanel from './ReviewPanel';
import VideoPanel from './VideoPanel';
import SequencePanel from './SequencePanel';
import ActiveLearningPanel from './ActiveLearningPanel';
import OcrActiveLearningPanel from './OcrActiveLearningPanel';
import OcrTrainingPanel from './OcrTrainingPanel';
import SegTrainingPanel from './SegTrainingPanel';
import './AnnotationWorkspace.css';
import { Sparkles, AlertTriangle, X, Upload, Image as ImageIcon, Check, ArrowLeft, ArrowRight, Brain, Rocket, Eye, Target, Tag, Package, Film, Undo2, Redo2, ZoomIn, ZoomOut, Maximize2, Trash2, ImageOff, Type, RotateCw, RotateCcw, Grid3x3, Wand2, Square, PenTool, RefreshCw, Scissors, Copy, ClipboardPaste, Info } from 'lucide-react';

import { API_URL } from '../config';

// Reports naturalWidth/naturalHeight once loaded so the parent can use the
// browser-corrected dimensions (EXIF orientation, etc.) for scale calculation.
const KonvaImage = ({ src, onLoad }) => {
    const [image] = useImage(src, 'anonymous');
    useEffect(() => {
        if (image && onLoad) {
            onLoad(image.naturalWidth || image.width, image.naturalHeight || image.height);
        }
    }, [image, onLoad]);
    return <Image image={image} />;
};

// ── Polygon/Polyline annotation shape ─────────────────────────────
// Renders a closed Line through ann.points (normalized) plus draggable
// vertex handles when selected. Dragging the fill/outline moves the whole
// shape; dragging a handle reshapes just that corner — the workflow for
// tracing a character's true rotated outline instead of an axis-aligned box.
const PolygonAnnotation = ({ ann, imgW, imgH, color, isSelected, isPanning, totalScale, onSelect, onRelabel, onMoveEnd, onVertexDragEnd, isDraggingShapeRef }) => {
    const lineRef = useRef(null);
    const pixelPoints = ann.points.map(([x, y]) => [x * imgW, y * imgH]);
    const flat = pixelPoints.flat();

    return (
        <Group
            draggable={isSelected && !isPanning}
            onDragStart={(e) => { if (e.target === e.currentTarget) isDraggingShapeRef.current = true; }}
            onDragEnd={(e) => {
                // Konva's drag events bubble — dragging a vertex Circle (below) also
                // fires this handler via bubbling. Only act when the Group itself
                // (not a child) was the node actually dragged, or a vertex drag would
                // get misread as a whole-shape move with a bogus offset and corrupt
                // every point (the annotation appears to "vanish").
                if (e.target !== e.currentTarget) return;
                isDraggingShapeRef.current = false;
                const dx = e.target.x();
                const dy = e.target.y();
                if (dx === 0 && dy === 0) return;
                e.target.x(0);
                e.target.y(0);
                onMoveEnd(pixelPoints.map(([x, y]) => [x + dx, y + dy]));
            }}
        >
            <Line
                ref={lineRef}
                points={flat}
                closed
                stroke={isSelected ? '#facc15' : color}
                strokeWidth={(isSelected ? 2 : 1.5) / totalScale}
                fill={ann.source === 'auto' ? 'rgba(167,139,250,0.07)' : 'rgba(244,63,94,0.07)'}
                onClick={onSelect}
                onTap={onSelect}
                onDblClick={onRelabel}
                onDblTap={onRelabel}
                onMouseEnter={e => { e.target.getStage().container().style.cursor = 'move'; }}
                onMouseLeave={e => { e.target.getStage().container().style.cursor = isPanning ? 'grab' : 'crosshair'; }}
            />
            {isSelected && pixelPoints.map(([x, y], i) => (
                <Circle
                    key={i}
                    x={x} y={y}
                    radius={5 / totalScale}
                    fill="#facc15"
                    stroke="#fff"
                    strokeWidth={1 / totalScale}
                    draggable={!isPanning}
                    onClick={(e) => { e.cancelBubble = true; }}
                    onTap={(e) => { e.cancelBubble = true; }}
                    onDragStart={(e) => { e.cancelBubble = true; isDraggingShapeRef.current = true; }}
                    onDragMove={(e) => {
                        e.cancelBubble = true; // stop the ancestor Group's onDragEnd from misfiring
                        const line = lineRef.current;
                        if (!line) return;
                        const pts = line.points().slice();
                        pts[i * 2] = e.target.x();
                        pts[i * 2 + 1] = e.target.y();
                        line.points(pts);
                        line.getLayer()?.batchDraw();
                    }}
                    onDragEnd={(e) => {
                        e.cancelBubble = true; // stop the ancestor Group's onDragEnd from misfiring
                        isDraggingShapeRef.current = false;
                        const line = lineRef.current;
                        const pts = line ? line.points() : flat;
                        const newPixelPoints = [];
                        for (let k = 0; k < pts.length; k += 2) newPixelPoints.push([pts[k], pts[k + 1]]);
                        onVertexDragEnd(newPixelPoints);
                    }}
                />
            ))}
        </Group>
    );
};

// ── Class Picker ────────────────────────────────────────────────
const ClassPicker = ({ classes, usedClasses, onConfirm, onCancel, remaining = 0, ocrMode = false }) => {
    const [customClass, setCustomClass] = useState('');
    const inputRef = useRef(null);

    // Merge project classes + already-used classes, dedupe, keep order
    const presetSet = new Set(classes);
    const usedOnly = usedClasses.filter(c => !presetSet.has(c));
    const allOptions = [...classes, ...usedOnly]; // presets first, then extra used ones

    useEffect(() => {
        if ((ocrMode || allOptions.length === 0) && inputRef.current) {
            inputRef.current.focus();
        }
    }, []); // eslint-disable-line

    // ── OCR mode: type the character (instant confirm) or click a key ──
    if (ocrMode) {
        const chars = [...new Set([...classes, ...usedOnly])];
        return (
            <div className="class-picker-overlay" onClick={onCancel}>
                <div className="class-picker class-picker--ocr" onClick={e => e.stopPropagation()}>
                    <div className="class-picker-header">
                        <span>
                            Which character is this?
                            {remaining > 0 && (
                                <span className="class-picker-counter"> — {remaining} left</span>
                            )}
                        </span>
                        <button className="class-picker-close" onClick={onCancel}><X size={16} /></button>
                    </div>
                    <input
                        ref={inputRef}
                        className="ocr-char-input"
                        placeholder="type it"
                        maxLength={1}
                        value=""
                        onChange={e => {
                            const ch = e.target.value.toUpperCase();
                            if (/^[0-9A-Z]$/.test(ch)) onConfirm(ch);
                        }}
                        onKeyDown={e => { if (e.key === 'Escape') onCancel(); }}
                    />
                    <p className="ocr-char-hint">Press the key on your keyboard — saves instantly. Or click below:</p>
                    <div className="ocr-char-grid">
                        {chars.map(cls => (
                            <button key={cls} className="ocr-char-key" onClick={() => onConfirm(cls)}>
                                {cls}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        );
    }

    const handleConfirmCustom = () => {
        if (customClass.trim()) onConfirm(customClass.trim());
    };

    return (
        <div className="class-picker-overlay" onClick={onCancel}>
            <div className="class-picker" onClick={e => e.stopPropagation()}>
                <div className="class-picker-header">
                    <span>
                        Select Class
                        {remaining > 0 && (
                            <span className="class-picker-counter"> — {remaining} left</span>
                        )}
                    </span>
                    <button className="class-picker-close" onClick={onCancel}><X size={16} /></button>
                </div>

                {allOptions.length > 0 && (
                    <div className="class-picker-list">
                        {classes.length > 0 && (
                            <p className="class-picker-section-label">Project Classes</p>
                        )}
                        {classes.map(cls => (
                            <button
                                key={cls}
                                className="class-picker-item"
                                onClick={() => onConfirm(cls)}
                            >
                                <span className="class-picker-dot" />
                                {cls}
                            </button>
                        ))}

                        {usedOnly.length > 0 && (
                            <>
                                <p className="class-picker-section-label class-picker-section-label--used">
                                    Recently Used
                                </p>
                                {usedOnly.map(cls => (
                                    <button
                                        key={cls}
                                        className="class-picker-item class-picker-item--used"
                                        onClick={() => onConfirm(cls)}
                                    >
                                        <span className="class-picker-dot class-picker-dot--used" />
                                        {cls}
                                    </button>
                                ))}
                            </>
                        )}
                    </div>
                )}

                <div className="class-picker-custom">
                    <input
                        ref={inputRef}
                        className="class-picker-input"
                        placeholder={allOptions.length > 0 ? 'Or type a new class…' : 'Type a class name…'}
                        value={customClass}
                        onChange={e => setCustomClass(e.target.value)}
                        onKeyDown={e => {
                            if (e.key === 'Enter') handleConfirmCustom();
                            if (e.key === 'Escape') onCancel();
                        }}
                    />
                    <button
                        className="class-picker-confirm"
                        onClick={handleConfirmCustom}
                        disabled={!customClass.trim()}
                    >
                        Add
                    </button>
                </div>
            </div>
        </div>
    );
};

// ── Main Workspace ──────────────────────────────────────────────
const AnnotationWorkspace = ({ project, onProjectUpdated }) => {
    const [images, setImages] = useState([]);
    const [annotations, setAnnotations] = useState([]);
    const [currentImage, setCurrentImage] = useState(null);
    // Actual displayed dimensions from the loaded image (may differ from backend
    // stored values when EXIF orientation rotates the image 90°/270°).
    const [loadedImageSize, setLoadedImageSize] = useState(null);
    const [imgVersion, setImgVersion] = useState({}); // image_id -> counter, cache-busts after rotation
    const [isDrawing, setIsDrawing] = useState(false);
    const [newAnnotation, setNewAnnotation] = useState(null);
    const [pendingAnnotation, setPendingAnnotation] = useState(null); // bbox waiting for class
    // ── Polyline drawing ── 'box' | 'polyline' — polyline traces a shape's
    // real (possibly rotated/skewed) outline instead of an axis-aligned box,
    // needed on angled plate photos where boxes of neighboring characters touch.
    const [drawMode, setDrawMode] = useState('box');
    const [newPolylinePoints, setNewPolylinePoints] = useState([]); // [{x,y}, ...] while drawing
    const [polylineCursor, setPolylineCursor] = useState(null); // rubber-band point to last click
    const [pendingPolyline, setPendingPolyline] = useState(null); // finished points waiting for class
    const [pendingShapeType, setPendingShapeType] = useState('polygon'); // 'polygon' (bbox-precision) | 'segment' (real mask)
    const [error, setError] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(null); // 0-100 during upload
    const [uploadFileCount, setUploadFileCount] = useState(0);
    const [isDragOver, setIsDragOver] = useState(false);
    const [statusMsg, setStatusMsg] = useState(null);
    const [canvasSize, setCanvasSize] = useState({ w: 800, h: 600 });
    const [allUsedClasses, setAllUsedClasses] = useState([]); // persists across image switches
    const [showTrainingPanel, setShowTrainingPanel] = useState(false);
    const [showAutoAnnotatePanel, setShowAutoAnnotatePanel] = useState(false);
    const [showMainTrainingPanel, setShowMainTrainingPanel] = useState(false);
    const [showLabelsPanel, setShowLabelsPanel] = useState(false);
    const [showModelsPanel, setShowModelsPanel] = useState(false);
    const [showReviewPanel, setShowReviewPanel] = useState(false);
    const [showVideoPanel, setShowVideoPanel] = useState(false);
    const [showSequencePanel, setShowSequencePanel] = useState(false);
    const [showActiveLearningPanel, setShowActiveLearningPanel] = useState(false);
    const [showOcrActiveLearningPanel, setShowOcrActiveLearningPanel] = useState(false);
    const [showOcrPanel, setShowOcrPanel] = useState(false);
    const [showSegPanel, setShowSegPanel] = useState(false);
    const [ocrAutoLabeling, setOcrAutoLabeling] = useState(false);
    const [seedModelInfo, setSeedModelInfo] = useState(null); // { exists, modified_at } — character detector status for OCR projects
    const [suggestedImageIds, setSuggestedImageIds] = useState(null);  // Set<id> or null (sidebar highlight)
    const [reviewFilterIds, setReviewFilterIds] = useState(null);      // Set<id> or null (ReviewPanel filter)
    const [reviewFilterLabel, setReviewFilterLabel] = useState(null);  // string or null — shown in ReviewPanel header
    const [reviewFilterClassName, setReviewFilterClassName] = useState(null); // raw class name — enables per-class bulk delete in ReviewPanel
    // Local copy of classes so edits from LabelsPanel are reflected instantly
    const [localClasses, setLocalClasses] = useState(project.classes || []);
    const [aiPrompt, setAiPrompt] = useState('');
    const [clearExisting, setClearExisting] = useState(false);
    const [isDetecting, setIsDetecting] = useState(false);
    const [classifyingAnnId, setClassifyingAnnId] = useState(null); // id of AI ann being classified
    const aiQueueRef = useRef([]); // queue of {id, bbox} waiting for ClassPicker
    const canvasAreaRef = useRef(null);
    const canvasCenterRef = useRef(null);

    // ── Bbox editing ─────────────────────────────────────────────
    const [selectedAnnId, setSelectedAnnId] = useState(null);
    const transformerRef = useRef(null);
    const annNodesRef = useRef({});

    // ── Copy / paste a shape (box, polygon or segment) ────────────
    const [clipboardAnn, setClipboardAnn] = useState(null);

    // Ignore wheel-zoom while a shape is actively being dragged — a hand
    // resting on a scroll wheel/trackpad mid-drag (very easy right after a
    // paste, since the mouse is already moving the new shape into place)
    // was firing handleWheel and snapping the whole canvas to a different
    // zoom level ("zoomed out at some random point") in the middle of the drag.
    const isDraggingShapeRef = useRef(false);
    // Konva only fires its own dragstart after the pointer has moved a few
    // px past mousedown — so a wheel nudge in that initial window (press,
    // then scroll before moving) slipped past isDraggingShapeRef above and
    // still zoomed. Track the raw mouse-button-down state too, from the
    // moment of press, so that gap is covered as well.
    const isPointerDownRef = useRef(false);
    useEffect(() => {
        const down = () => { isPointerDownRef.current = true; };
        const up = () => { isPointerDownRef.current = false; };
        window.addEventListener('mousedown', down);
        window.addEventListener('mouseup', up);
        window.addEventListener('touchstart', down);
        window.addEventListener('touchend', up);
        return () => {
            window.removeEventListener('mousedown', down);
            window.removeEventListener('mouseup', up);
            window.removeEventListener('touchstart', down);
            window.removeEventListener('touchend', up);
        };
    }, []);

    // ── Grid overlay + fine rotation ─────────────────────────────
    const [showGrid, setShowGrid] = useState(false);
    const [gridSpacing, setGridSpacing] = useState(50); // image pixels between lines
    const [previewAngle, setPreviewAngle] = useState(0); // live rotation preview (degrees, CW+)
    const [isStraightening, setIsStraightening] = useState(false);

    // ── Zoom / pan ───────────────────────────────────────────────
    const [userZoom, setUserZoom] = useState(1);
    const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState(false);

    // ── Undo / redo ──────────────────────────────────────────────
    const [history, setHistory] = useState([]);
    const [redoStack, setRedoStack] = useState([]);

    // Declared early so handlers defined below can reference them
    const imgW = loadedImageSize?.width  || currentImage?.width  || 1;
    const imgH = loadedImageSize?.height || currentImage?.height || 1;
    const scale = currentImage ? Math.min(1, canvasSize.w / imgW, canvasSize.h / imgH) : 1;
    const stageW = currentImage ? Math.round(imgW * scale) : canvasSize.w;
    const stageH = currentImage ? Math.round(imgH * scale) : canvasSize.h;

    const pollTask = useCallback((taskId, onComplete, onProgress) => {
        const interval = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/pipeline/task-status/${taskId}`);
                if (res.data.status === 'SUCCESS') {
                    clearInterval(interval);
                    onComplete(res.data.result);
                } else if (res.data.status === 'FAILURE') {
                    clearInterval(interval);
                    setError("AI Detection failed: " + (res.data.error || "Unknown error"));
                    setIsDetecting(false);
                } else if (onProgress && res.data.meta) {
                    onProgress(res.data.meta);
                }
            } catch (e) {
                clearInterval(interval);
                setIsDetecting(false);
            }
        }, 2000);
        return interval;
    }, []);

    // Open ClassPicker for the next AI annotation in the queue
    const processNextAIAnnotation = useCallback((img) => {
        const queue = aiQueueRef.current;
        if (queue.length === 0) { setClassifyingAnnId(null); return; }
        const next = queue.shift();
        const image = img || currentImage;
        if (!image) return;
        const bx = (next.bbox[0] - next.bbox[2] / 2) * image.width;
        const by = (next.bbox[1] - next.bbox[3] / 2) * image.height;
        const bw = next.bbox[2] * image.width;
        const bh = next.bbox[3] * image.height;
        setClassifyingAnnId(next.id);
        setPendingAnnotation({ x: bx, y: by, width: bw, height: bh });
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [currentImage]);

    // ── History helpers ──────────────────────────────────────────
    const pushHistory = useCallback((action) => {
        setHistory(prev => [...prev.slice(-49), action]);
        setRedoStack([]);
    }, []);

    const handleUndo = useCallback(async () => {
        setHistory(prev => {
            if (!prev.length) return prev;
            const action = prev[prev.length - 1];
            const next = prev.slice(0, -1);
            setRedoStack(r => [...r, action]);
            (async () => {
                try {
                    if (action.type === 'add') {
                        await axios.delete(`${API_URL}/annotations/${action.ann.id}`);
                        setAnnotations(a => a.filter(x => x.id !== action.ann.id));
                    } else if (action.type === 'delete') {
                        const res = await axios.post(`${API_URL}/annotations`, {
                            image_id: action.ann.image_id, class_name: action.ann.class_name,
                            bbox: action.ann.bbox, source: action.ann.source,
                            annotation_type: action.ann.annotation_type, points: action.ann.points,
                        });
                        setAnnotations(a => [...a, res.data]);
                    } else if (action.type === 'bbox') {
                        await axios.patch(`${API_URL}/annotations/${action.id}`, { bbox: action.oldBbox });
                        setAnnotations(a => a.map(x => x.id === action.id ? { ...x, bbox: action.oldBbox } : x));
                    } else if (action.type === 'points') {
                        await axios.patch(`${API_URL}/annotations/${action.id}/points`, { points: action.oldPoints });
                        setAnnotations(a => a.map(x => x.id === action.id ? { ...x, points: action.oldPoints } : x));
                    }
                } catch { /* ignore */ }
            })();
            return next;
        });
    }, []);

    const handleRedo = useCallback(async () => {
        setRedoStack(prev => {
            if (!prev.length) return prev;
            const action = prev[prev.length - 1];
            const next = prev.slice(0, -1);
            setHistory(h => [...h, action]);
            (async () => {
                try {
                    if (action.type === 'add') {
                        const res = await axios.post(`${API_URL}/annotations`, {
                            image_id: action.ann.image_id, class_name: action.ann.class_name,
                            bbox: action.ann.bbox, source: action.ann.source,
                            annotation_type: action.ann.annotation_type, points: action.ann.points,
                        });
                        setAnnotations(a => [...a, res.data]);
                    } else if (action.type === 'delete') {
                        await axios.delete(`${API_URL}/annotations/${action.ann.id}`);
                        setAnnotations(a => a.filter(x => x.id !== action.ann.id));
                    } else if (action.type === 'bbox') {
                        await axios.patch(`${API_URL}/annotations/${action.id}`, { bbox: action.newBbox });
                        setAnnotations(a => a.map(x => x.id === action.id ? { ...x, bbox: action.newBbox } : x));
                    } else if (action.type === 'points') {
                        await axios.patch(`${API_URL}/annotations/${action.id}/points`, { points: action.newPoints });
                        setAnnotations(a => a.map(x => x.id === action.id ? { ...x, points: action.newPoints } : x));
                    }
                } catch { /* ignore */ }
            })();
            return next;
        });
    }, []);

    // ── Annotation selection / transform ─────────────────────────
    const handleAnnClick = useCallback((annId, e) => {
        if (e) e.cancelBubble = true;
        setSelectedAnnId(annId);
    }, []);

    // Double-click any annotation (box or polyline) to fix a wrong label —
    // reuses the same ClassPicker + PATCH .../classify flow already used for
    // unclassified AI boxes, just opened for an annotation that already has a class.
    const handleEditLabel = useCallback((ann, e) => {
        if (e) e.cancelBubble = true;
        if (isPanning) return;
        setSelectedAnnId(null);
        setClassifyingAnnId(ann.id);
        if ((ann.annotation_type === 'polygon' || ann.annotation_type === 'segment') && ann.points) {
            setPendingPolyline(ann.points.map(([x, y]) => ({ x: x * imgW, y: y * imgH })));
        } else {
            setPendingAnnotation({
                x: (ann.bbox[0] - ann.bbox[2] / 2) * imgW,
                y: (ann.bbox[1] - ann.bbox[3] / 2) * imgH,
                width: ann.bbox[2] * imgW,
                height: ann.bbox[3] * imgH,
            });
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isPanning, imgW, imgH]);

    const handleAnnDragEnd = useCallback((e, ann) => {
        const node = e.target;
        const newX = node.x();
        const newY = node.y();
        const bw = ann.bbox[2] * imgW;
        const bh = ann.bbox[3] * imgH;
        const newBbox = [
            (newX + bw / 2) / imgW,
            (newY + bh / 2) / imgH,
            ann.bbox[2],
            ann.bbox[3],
        ];
        pushHistory({ type: 'bbox', id: ann.id, oldBbox: ann.bbox, newBbox });
        setSelectedAnnId(null);
        axios.patch(`${API_URL}/annotations/${ann.id}`, { bbox: newBbox })
            .then(res => setAnnotations(prev => prev.map(a => a.id === ann.id ? res.data : a)))
            .catch(() => setError('Failed to move annotation.'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imgW, imgH, pushHistory]);

    const handleAnnTransformEnd = useCallback((e, ann) => {
        const node = e.target;
        const scaleX = node.scaleX();
        const scaleY = node.scaleY();
        const newW = Math.max(5, node.width() * scaleX);
        const newH = Math.max(5, node.height() * scaleY);
        const newX = node.x();
        const newY = node.y();
        node.scaleX(1);
        node.scaleY(1);
        node.width(newW);
        node.height(newH);
        const newBbox = [
            (newX + newW / 2) / imgW,
            (newY + newH / 2) / imgH,
            newW / imgW,
            newH / imgH,
        ];
        pushHistory({ type: 'bbox', id: ann.id, oldBbox: ann.bbox, newBbox });
        setSelectedAnnId(null);
        axios.patch(`${API_URL}/annotations/${ann.id}`, { bbox: newBbox })
            .then(res => setAnnotations(prev => prev.map(a => a.id === ann.id ? res.data : a)))
            .catch(() => setError('Failed to resize annotation.'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imgW, imgH, pushHistory]);

    // Whole-shape drag of a polygon/polyline annotation
    const handlePolygonMoveEnd = useCallback((ann, newPixelPoints) => {
        const newPoints = newPixelPoints.map(([x, y]) => [x / imgW, y / imgH]);
        pushHistory({ type: 'points', id: ann.id, oldPoints: ann.points, newPoints });
        axios.patch(`${API_URL}/annotations/${ann.id}/points`, { points: newPoints })
            .then(res => setAnnotations(prev => prev.map(a => a.id === ann.id ? res.data : a)))
            .catch(() => setError('Failed to move annotation.'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imgW, imgH, pushHistory]);

    // Single-vertex drag — reshapes the polygon to hug the true character outline.
    // Selection is intentionally kept so corners can be nudged one after another.
    const handlePolygonVertexEnd = useCallback((ann, newPixelPoints) => {
        const newPoints = newPixelPoints.map(([x, y]) => [x / imgW, y / imgH]);
        pushHistory({ type: 'points', id: ann.id, oldPoints: ann.points, newPoints });
        axios.patch(`${API_URL}/annotations/${ann.id}/points`, { points: newPoints })
            .then(res => setAnnotations(prev => prev.map(a => a.id === ann.id ? res.data : a)))
            .catch(() => setError('Failed to reshape annotation.'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imgW, imgH, pushHistory]);

    // ── Zoom via mouse wheel ──────────────────────────────────────
    const handleWheel = useCallback((e) => {
        e.evt.preventDefault();
        if (isDraggingShapeRef.current || isPointerDownRef.current) return; // don't fight an in-progress shape drag (or a press about to become one)
        const stage = e.target.getStage();
        const pointer = stage.getPointerPosition();
        const scaleBy = 1.12;
        const oldZoom = userZoom;
        const newZoom = Math.max(0.2, Math.min(15, e.evt.deltaY < 0 ? oldZoom * scaleBy : oldZoom / scaleBy));
        // Total stage x = centerOffset + panOffset; derive the content point under the cursor
        const cx = Math.round((canvasSize.w - stageW) / 2);
        const cy = Math.round((canvasSize.h - stageH) / 2);
        const totalX = cx + stagePos.x;
        const totalY = cy + stagePos.y;
        const mousePointTo = {
            x: (pointer.x - totalX) / (scale * oldZoom),
            y: (pointer.y - totalY) / (scale * oldZoom),
        };
        setUserZoom(newZoom);
        // New panOffset = newTotalX - cx; where newTotalX keeps the hovered point fixed
        setStagePos({
            x: pointer.x - mousePointTo.x * scale * newZoom - cx,
            y: pointer.y - mousePointTo.y * scale * newZoom - cy,
        });
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [userZoom, stagePos, scale, canvasSize, stageW, stageH]);

    const resetZoom = useCallback(() => {
        setUserZoom(1);
        setStagePos({ x: 0, y: 0 });
    }, []);

    const handleAIDetect = async () => {
        if (!currentImage || !aiPrompt.trim()) return;
        setIsDetecting(true);
        const img = currentImage;
        try {
            const res = await axios.post(`${API_URL}/pipeline/ai-prompt`, {
                project_id: project.id,
                image_id: img.id,
                prompt: aiPrompt.trim(),
                clear_existing: clearExisting
            });
            pollTask(res.data.task_id, (result) => {
                setIsDetecting(false);
                if (result.count > 0) {
                    showStatus(`✓ Found ${result.count} object${result.count !== 1 ? 's' : ''} — assign classes below`);
                    // Fetch fresh annotations then queue unclassified ones for ClassPicker
                    axios.get(`${API_URL}/annotations/image/${img.id}`).then(r => {
                        setAnnotations(r.data);
                        const unclassified = r.data.filter(a => a.source === 'ai_prompt' && !a.class_name);
                        if (unclassified.length > 0) {
                            aiQueueRef.current = unclassified.map(a => ({ id: a.id, bbox: a.bbox }));
                            processNextAIAnnotation(img);
                        }
                    });
                } else {
                    showStatus("No objects found.");
                }
            });
        } catch (e) {
            setError("Failed to start AI detection.");
            setIsDetecting(false);
        }
    };

    const handleAIApplyAll = async () => {
        if (!aiPrompt.trim()) return;
        const count = images.filter(img => img.status === 'pending').length;
        if (count === 0) {
            showStatus("No pending images to annotate.");
            return;
        }
        
        const confirmMsg = `AI Plan:
- Use prompt: "${aiPrompt.trim()}"
- Process ${count} pending images
- Detected boxes will be added to the images

Do you want to proceed?`;

        if (!window.confirm(confirmMsg)) return;

        setIsDetecting(true);
        try {
            const res = await axios.post(`${API_URL}/pipeline/ai-bulk-prompt`, {
                project_id: project.id,
                prompt: aiPrompt.trim()
            });
            pollTask(res.data.task_id, (result) => {
                setIsDetecting(false);
                showStatus(`✓ AI Bulk complete: ${result.total_found} objects found across ${result.processed} images.`);
                // Refresh list
                axios.get(`${API_URL}/images/project/${project.id}`).then(r => setImages(r.data));
                if (currentImage) handleImageClick(currentImage);
            }, (progress) => {
                if (progress.current) {
                    setStatusMsg(`AI Processing: ${progress.current} / ${progress.total}...`);
                }
            });
        } catch (e) {
            setError("Failed to start bulk AI detection.");
            setIsDetecting(false);
        }
    };

    const handleAcceptAnnotation = async (annId) => {
        try {
            await axios.patch(`${API_URL}/annotations/${annId}/verify`);
            setAnnotations(prev => prev.map(a => a.id === annId ? { ...a, source: 'manual' } : a));
        } catch (e) {
            setError("Failed to verify annotation.");
        }
    };

    const handleRejectAnnotation = async (annId) => {
        const ann = annotations.find(a => a.id === annId);
        try {
            await axios.delete(`${API_URL}/annotations/${annId}`);
            if (ann) pushHistory({ type: 'delete', ann });
            setAnnotations(prev => prev.filter(a => a.id !== annId));
            if (selectedAnnId === annId) setSelectedAnnId(null);
        } catch (e) {
            setError("Failed to delete annotation.");
        }
    };

    // Copy the selected shape (box, polygon or segment mask) so it can be
    // pasted again — same class + same exact outline, no redrawing.
    const handleCopyAnnotation = useCallback(() => {
        const ann = annotations.find(a => a.id === selectedAnnId);
        if (!ann) return;
        setClipboardAnn(ann);
        showStatus(`Copied ${ann.class_name} — Ctrl+V (or the Paste button) to place it`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [annotations, selectedAnnId]);

    // Paste the copied shape as a new annotation, nudged so it doesn't sit
    // exactly on top of the original — drag it into place afterward.
    const handlePasteAnnotation = useCallback(() => {
        if (!clipboardAnn || !currentImage) return;
        const dxN = 24 / imgW;
        const dyN = 24 / imgH;
        const isShape = (clipboardAnn.annotation_type === 'polygon' || clipboardAnn.annotation_type === 'segment') && clipboardAnn.points;
        const body = {
            image_id: currentImage.id,
            class_name: clipboardAnn.class_name,
            annotation_type: clipboardAnn.annotation_type,
            ...(isShape
                ? { points: clipboardAnn.points.map(([x, y]) => [x + dxN, y + dyN]) }
                : { bbox: [clipboardAnn.bbox[0] + dxN, clipboardAnn.bbox[1] + dyN, clipboardAnn.bbox[2], clipboardAnn.bbox[3]] }),
        };
        axios.post(`${API_URL}/annotations`, body)
            .then(res => {
                setAnnotations(prev => [...prev, res.data]);
                pushHistory({ type: 'add', ann: res.data });
                setSelectedAnnId(res.data.id);
                setImages(prev => prev.map(img =>
                    img.id === currentImage.id ? { ...img, status: 'annotated' } : img
                ));
                showStatus(`Pasted ${clipboardAnn.class_name} — drag it into place`);
            })
            .catch(() => setError('Failed to paste annotation.'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [clipboardAnn, currentImage, imgW, imgH, pushHistory]);

    const handleDeleteImage = async () => {
        if (!currentImage) return;
        if (!window.confirm(`Delete "${currentImage.filename}"?\n\nThis will permanently remove the image and all its annotations.`)) return;
        try {
            await axios.delete(`${API_URL}/images/${currentImage.id}`);
            const remaining = images.filter(img => img.id !== currentImage.id);
            setImages(remaining);
            // Navigate to next available image or clear canvas
            const idx = images.findIndex(img => img.id === currentImage.id);
            const next = remaining[idx] || remaining[idx - 1] || null;
            if (next) {
                handleImageClick(next);
            } else {
                setCurrentImage(null);
                setAnnotations([]);
            }
            showStatus('Image deleted');
        } catch (e) {
            setError('Failed to delete image.');
        }
    };

    const handleClearAllAnnotations = async () => {
        if (!currentImage || annotations.length === 0) return;
        if (!window.confirm(`Delete all ${annotations.length} annotation(s) on this image?`)) return;
        try {
            await Promise.all(annotations.map(a => axios.delete(`${API_URL}/annotations/${a.id}`)));
            setAnnotations([]);
            setSelectedAnnId(null);
            setHistory([]);
            setRedoStack([]);
        } catch (e) {
            setError('Failed to clear annotations.');
        }
    };

    const handleAcceptAll = async () => {
        const toVerify = annotations.filter(a => a.source !== 'manual');
        if (toVerify.length === 0) return;
        
        try {
            // Sequential to be safe, or could do Promise.all
            await Promise.all(toVerify.map(a => axios.patch(`${API_URL}/annotations/${a.id}/verify`)));
            setAnnotations(prev => prev.map(a => ({ ...a, source: 'manual' })));
            showStatus(`✓ Accepted ${toVerify.length} annotations`);
        } catch (e) {
            setError("Failed to accept all annotations.");
        }
    };

    const handleMarkEmpty = async () => {
        if (!currentImage) return;
        try {
            await axios.patch(`${API_URL}/images/${currentImage.id}/mark-empty`);
            // Update sidebar status
            setImages(prev => prev.map(img =>
                img.id === currentImage.id ? { ...img, status: 'annotated' } : img
            ));
            showStatus('Marked as no objects — frame skipped.');
            // Auto-advance to next pending image
            const nextPending = images.find(
                img => img.id !== currentImage.id && img.status === 'pending'
            );
            if (nextPending) handleImageClick(nextPending);
        } catch {
            setError('Failed to mark image as empty.');
        }
    };

    const fileInputRef = useRef(null);
    const datasetImportRef = useRef(null);
    const [exportingDataset, setExportingDataset] = useState(false);
    const [exportProgress, setExportProgress] = useState(null);
    const [importingDataset, setImportingDataset] = useState(false);

    useEffect(() => {
        if (project) {
            axios.get(`${API_URL}/images/project/${project.id}`)
                .then(res => setImages(res.data))
                .catch(() => setError("Failed to load images."));
        }
    }, [project]);

    // Re-sync localClasses whenever the active project changes (e.g. user
    // navigates back to dashboard then reopens the project — ProjectList may
    // return stale data that doesn't yet include classes added this session).
    useEffect(() => {
        axios.get(`${API_URL}/projects/${project.id}`)
            .then(res => setLocalClasses(res.data.classes || []))
            .catch(() => setLocalClasses(project.classes || []));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [project.id]);

    const showStatus = (msg) => {
        setStatusMsg(msg);
        setTimeout(() => setStatusMsg(null), 3500);
    };

    // Character detector (seed YOLO model) status — shown in the OCR sidebar
    // so it's obvious whether "find the char boxes" step has been trained yet.
    const loadSeedModelInfo = useCallback(() => {
        axios.get(`${API_URL}/pipeline/model-details/${project.id}`)
            .then(res => setSeedModelInfo(res.data.seed || null))
            .catch(() => {});
    }, [project.id]);

    useEffect(() => {
        if (project.project_type === 'ocr' || project.project_type === 'combined') loadSeedModelInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [project.id, project.project_type]);

    // Called from ActiveLearningPanel when user clicks "Annotate These"
    const handleAnnotateImages = (imageIds) => {
        const idSet = new Set(imageIds.map(String));
        setSuggestedImageIds(idSet);   // highlight in sidebar
        setReviewFilterIds(idSet);     // open in ReviewPanel
        setShowReviewPanel(true);
    };

    // Called from LabelsPanel's "edit images" action — jump into ReviewPanel
    // showing only the images that use a specific class, so a class that's
    // failing detection can be renamed, re-boxed, or reassigned without
    // hunting through the whole project.
    const handleEditClassImages = (className, imageIds) => {
        const idSet = new Set(imageIds.map(String));
        setReviewFilterIds(idSet);
        setReviewFilterLabel(`Class "${className}" — ${idSet.size} image${idSet.size !== 1 ? 's' : ''}`);
        setReviewFilterClassName(className);
        setShowLabelsPanel(false);
        setShowReviewPanel(true);
    };

    const handleOcrAutoLabel = async () => {
        setOcrAutoLabeling(true);
        const shapeForMode = drawMode === 'segment' ? 'segment' : drawMode === 'polyline' ? 'polygon' : 'bbox';
        try {
            const res = await axios.post(`${API_URL}/ocr/auto-annotate/${project.id}`, {
                shape: shapeForMode,
            });
            showStatus(`✓ ${res.data.detail || `Pre-labeled ${res.data.labeled} characters.`}`);
            // Refresh image list + current image's annotations
            const imgRes = await axios.get(`${API_URL}/images/project/${project.id}`);
            setImages(imgRes.data);
            if (currentImage) {
                const annRes = await axios.get(`${API_URL}/annotations/image/${currentImage.id}`);
                setAnnotations(annRes.data);
            }
        } catch (e) {
            setError(e.response?.data?.detail || (shapeForMode === 'segment'
                ? 'Auto-labeling failed. Train a segmentation model first.'
                : 'Auto-labeling failed. Train the seed (or main) model first.'));
        } finally {
            setOcrAutoLabeling(false);
        }
    };

    // Shared refresh after any server-side rotation (file changed on disk,
    // boxes remapped server-side — reload image, annotations, reset undo/zoom)
    const refreshAfterRotation = async (imageId, width, height) => {
        const updated = { ...currentImage, width, height };
        setCurrentImage(updated);
        setImages(prev => prev.map(im => (im.id === updated.id ? updated : im)));
        setLoadedImageSize(null);
        setImgVersion(prev => ({ ...prev, [imageId]: (prev[imageId] || 0) + 1 }));
        setHistory([]);
        setRedoStack([]);
        setSelectedAnnId(null);
        resetZoom();
        const annRes = await axios.get(`${API_URL}/annotations/image/${imageId}`);
        setAnnotations(annRes.data);
    };

    const handleRotateImage = async (direction) => {
        if (!currentImage) return;
        try {
            const res = await axios.post(
                `${API_URL}/images/${currentImage.id}/rotate?direction=${direction}`
            );
            await refreshAfterRotation(currentImage.id, res.data.width, res.data.height);
            showStatus(`✓ Rotated ${direction === 'cw' ? 'clockwise' : 'counter-clockwise'}`);
        } catch {
            setError('Failed to rotate image.');
        }
    };

    // Bake the previewed slider angle into the file on the server
    const handleApplyPreviewRotation = async () => {
        if (!currentImage) return;
        const angle = Math.round(previewAngle * 10) / 10;
        if (!angle) { setPreviewAngle(0); return; }
        try {
            const res = await axios.post(
                `${API_URL}/images/${currentImage.id}/rotate-fine?angle=${angle}`
            );
            setPreviewAngle(0);
            await refreshAfterRotation(currentImage.id, res.data.width, res.data.height);
            showStatus(`✓ Rotated ${angle > 0 ? '+' : ''}${angle}°`);
        } catch {
            setError('Failed to rotate image.');
        }
    };

    const handleAutoStraighten = async () => {
        if (!currentImage || isStraightening) return;
        setIsStraightening(true);
        try {
            const res = await axios.post(
                `${API_URL}/images/${currentImage.id}/auto-straighten`
            );
            if (res.data.status === 'already_straight') {
                showStatus('Image already looks straight — no tilt detected.');
            } else {
                await refreshAfterRotation(currentImage.id, res.data.width, res.data.height);
                showStatus(`✓ Auto-straightened by ${res.data.angle.toFixed(1)}°`);
            }
        } catch (e) {
            setError(e.response?.data?.detail || 'Auto-straighten failed.');
        } finally {
            setIsStraightening(false);
        }
    };

    const handleImageLoad = useCallback((w, h) => {
        setLoadedImageSize({ width: w, height: h });
    }, []);

    const handleNavigateImage = (direction) => {
        if (!currentImage || images.length === 0) return;
        const idx = images.findIndex(img => img.id === currentImage.id);
        if (idx === -1) return;
        const nextIdx = direction === 'next' ? idx + 1 : idx - 1;
        if (nextIdx < 0 || nextIdx >= images.length) return;
        handleImageClick(images[nextIdx]);
    };

    const handleImageClick = (image) => {
        setCurrentImage(image);
        setLoadedImageSize(null); // reset until new image loads
        setPendingAnnotation(null);
        setNewAnnotation(null);
        axios.get(`${API_URL}/annotations/image/${image.id}`)
            .then(res => {
                setAnnotations(res.data);
                // Merge any classes from this image into the workspace-wide used list (skip empty)
                const named = res.data.map(a => a.class_name).filter(Boolean);
                if (named.length > 0) {
                    setAllUsedClasses(prev => [...new Set([...prev, ...named])]);
                }
            })
            .catch(() => setAnnotations([]));
    };

    // Core upload logic — accepts a plain JS array of File objects
    const uploadFiles = (files) => {
        if (!files.length) return;

        const formData = new FormData();
        for (const file of files) {
            formData.append("files", file);
        }

        setUploading(true);
        setUploadProgress(0);
        setUploadFileCount(files.length);

        // Do NOT set Content-Type — axios detects FormData and lets the
        // browser attach the correct multipart boundary automatically.
        axios.post(`${API_URL}/images/upload/${project.id}`, formData, {
            onUploadProgress: (evt) => {
                if (evt.total) {
                    setUploadProgress(Math.round((evt.loaded / evt.total) * 100));
                }
            },
        })
            .then(res => {
                const uploaded = res.data.uploaded ?? res.data;
                const failed = res.data.failed ?? [];
                setImages(prev => [...prev, ...uploaded]);
                if (uploaded.length > 0)
                    showStatus(`✓ ${uploaded.length} image${uploaded.length !== 1 ? 's' : ''} uploaded`);
                if (failed.length > 0)
                    showStatus(`⚠ ${failed.length} file${failed.length !== 1 ? 's' : ''} skipped: ${failed.map(f => f.filename).join(', ')}`);
            })
            .catch((err) => {
                const detail = err.response?.data?.detail;
                setError(detail ? String(detail) : "Upload failed. Please try again.");
            })
            .finally(() => {
                setUploading(false);
                setUploadProgress(null);
                setUploadFileCount(0);
                // Reset input AFTER request completes so File objects stay readable
                if (fileInputRef.current) fileInputRef.current.value = '';
            });
    };

    // File input onChange
    const handleFileUpload = (e) => {
        const selected = Array.from(e.target.files || []);
        // Don't clear input value here — do it in .finally() above
        uploadFiles(selected);
    };

    // ── Dataset export/import — portable images+annotations zip, so a
    // labeled dataset (including segment masks) can move into a different
    // project (e.g. one built around SAM) without redrawing anything. ──
    const handleExportDataset = async () => {
        setExportingDataset(true);
        setExportProgress('Requesting…');
        try {
            const res = await axios.get(`${API_URL}/images/export/${project.id}`, {
                responseType: 'blob',
                onDownloadProgress: (evt) => {
                    // Content-Length usually isn't set on a streamed zip, so evt.total
                    // is 0 — show bytes received so far instead of a % that can't be computed.
                    const mb = (evt.loaded / (1024 * 1024)).toFixed(1);
                    setExportProgress(evt.total
                        ? `${Math.round((evt.loaded / evt.total) * 100)}%`
                        : `${mb} MB…`);
                },
            });
            const url = window.URL.createObjectURL(new Blob([res.data]));
            const a = document.createElement('a');
            a.href = url;
            a.download = `${project.name.replace(/[^A-Za-z0-9]+/g, '_').toLowerCase() || 'project'}_dataset.zip`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            showStatus('✓ Dataset downloaded');
        } catch (err) {
            // responseType:'blob' means an error body also arrives as a Blob,
            // not parsed JSON — err.response.data.detail is always undefined
            // unless we read the Blob's text ourselves.
            let detail = 'Failed to export dataset.';
            if (err.response?.data instanceof Blob) {
                try {
                    const text = await err.response.data.text();
                    detail = JSON.parse(text)?.detail || detail;
                } catch { /* not JSON — keep generic message */ }
            } else if (err.response?.data?.detail) {
                detail = err.response.data.detail;
            }
            setError(detail);
        } finally {
            setExportingDataset(false);
            setExportProgress(null);
        }
    };

    const handleImportDataset = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setImportingDataset(true);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const res = await axios.post(`${API_URL}/images/import/${project.id}`, formData);
            const { images_imported, annotations_imported, skipped } = res.data;
            showStatus(`✓ Imported ${images_imported} image${images_imported !== 1 ? 's' : ''}, ${annotations_imported} annotation${annotations_imported !== 1 ? 's' : ''}${skipped?.length ? ` (${skipped.length} skipped)` : ''}`);
            const [imgRes] = await Promise.all([
                axios.get(`${API_URL}/images/project/${project.id}`),
            ]);
            setImages(imgRes.data);
            if (res.data.classes) setLocalClasses(res.data.classes);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to import dataset.');
        } finally {
            setImportingDataset(false);
            if (datasetImportRef.current) datasetImportRef.current.value = '';
        }
    };

    // Drag & drop handlers on the image list
    const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragOver(true); };
    const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragOver(false); };
    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);
        if (uploading) return;
        const dropped = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (dropped.length) uploadFiles(dropped);
    };

    const handleMouseDown = (e) => {
        if (isPanning) return;
        if (pendingAnnotation || pendingPolyline) return;
        if (previewAngle !== 0) return; // rotation preview active — apply or reset first
        // Only block when clicking on a visible annotation shape (Rect/Line/Circle/Group/Text)
        // Stage and Layer nodes are safe to draw on; KonvaImage has listening=false
        const cls = e.target.getClassName ? e.target.getClassName() : '';
        if (cls === 'Rect' || cls === 'Group' || cls === 'Text' || cls === 'Line' || cls === 'Circle') return;
        setSelectedAnnId(null);
        const pos = e.target.getStage().getRelativePointerPosition();

        if (drawMode === 'polyline' || drawMode === 'segment') {
            setNewPolylinePoints(prev => [...prev, pos]);
            return;
        }
        setIsDrawing(true);
        setNewAnnotation({ x: pos.x, y: pos.y, width: 0, height: 0 });
    };

    const handleMouseMove = (e) => {
        if (drawMode === 'polyline' || drawMode === 'segment') {
            if (newPolylinePoints.length === 0) return;
            setPolylineCursor(e.target.getStage().getRelativePointerPosition());
            return;
        }
        if (!isDrawing) return;
        const pos = e.target.getStage().getRelativePointerPosition();
        setNewAnnotation(prev => ({
            ...prev,
            width: pos.x - prev.x,
            height: pos.y - prev.y,
        }));
    };

    const handleMouseUp = () => {
        if (drawMode === 'polyline' || drawMode === 'segment') return; // finished via double-click / Enter
        setIsDrawing(false);
        if (!newAnnotation || Math.abs(newAnnotation.width) < 5 || Math.abs(newAnnotation.height) < 5) {
            setNewAnnotation(null);
            return;
        }
        // Normalise so x/y is always top-left
        const x = newAnnotation.width < 0 ? newAnnotation.x + newAnnotation.width : newAnnotation.x;
        const y = newAnnotation.height < 0 ? newAnnotation.y + newAnnotation.height : newAnnotation.y;
        const w = Math.abs(newAnnotation.width);
        const h = Math.abs(newAnnotation.height);
        setPendingAnnotation({ x, y, width: w, height: h });
    };

    // Switching tools mid-draw discards whatever was in progress
    const changeDrawMode = (mode) => {
        setDrawMode(mode);
        setIsDrawing(false);
        setNewAnnotation(null);
        setNewPolylinePoints([]);
        setPolylineCursor(null);
    };

    const finishPolyline = useCallback(() => {
        if (newPolylinePoints.length < 3) {
            setNewPolylinePoints([]);
            setPolylineCursor(null);
            return;
        }
        setPendingPolyline(newPolylinePoints);
        setPendingShapeType(drawMode === 'segment' ? 'segment' : 'polygon');
        setNewPolylinePoints([]);
        setPolylineCursor(null);
    }, [newPolylinePoints, drawMode]);

    const cancelPolyline = useCallback(() => {
        setNewPolylinePoints([]);
        setPolylineCursor(null);
    }, []);

    const handleConvertToPolyline = async () => {
        if (!currentImage) return;
        const boxCount = annotations.filter(a => !['polygon', 'segment'].includes(a.annotation_type)).length;
        if (boxCount === 0) { showStatus('No box annotations left to convert.'); return; }
        try {
            const res = await axios.patch(`${API_URL}/annotations/image/${currentImage.id}/convert-to-polygon`);
            setAnnotations(res.data);
            setSelectedAnnId(null);
            showStatus(`✓ Converted ${boxCount} box annotation${boxCount !== 1 ? 's' : ''} to polylines — drag the corners to fit`);
        } catch {
            setError('Failed to convert annotations to polylines.');
        }
    };

    const handleClassConfirm = (className) => {
        const ann = pendingAnnotation;
        const polyline = pendingPolyline;
        const shapeType = pendingShapeType;
        const annId = classifyingAnnId;
        setPendingAnnotation(null);
        setPendingPolyline(null);
        setPendingShapeType('polygon');
        setNewAnnotation(null);
        setClassifyingAnnId(null);

        if (polyline && !annId) {
            // New drawn polyline/segment — POST it with normalized points (backend derives the AABB bbox)
            const points = polyline.map(p => [p.x / imgW, p.y / imgH]);
            axios.post(`${API_URL}/annotations`, {
                image_id: currentImage.id,
                class_name: className,
                annotation_type: shapeType,
                points,
            })
                .then(res => {
                    setAnnotations(prev => [...prev, res.data]);
                    pushHistory({ type: 'add', ann: res.data });
                    setImages(prev => prev.map(img =>
                        img.id === currentImage.id ? { ...img, status: 'annotated' } : img
                    ));
                    setAllUsedClasses(prev =>
                        prev.includes(className) ? prev : [...prev, className]
                    );
                    showStatus(`Annotation added: ${className}`);
                })
                .catch(() => setError("Failed to save annotation."));
            return;
        }

        if (annId) {
            // Classifying an existing AI-detected annotation — PATCH it
            axios.patch(`${API_URL}/annotations/${annId}/classify`, { class_name: className })
                .then(res => {
                    setAnnotations(prev => prev.map(a => a.id === annId ? res.data : a));
                    setAllUsedClasses(prev => prev.includes(className) ? prev : [...prev, className]);
                    showStatus(`Classified: ${className}`);
                    processNextAIAnnotation();
                })
                .catch(() => {
                    setError("Failed to save class.");
                    processNextAIAnnotation();
                });
        } else {
            // New drawn annotation — POST it
            const bbox = [
                (ann.x + ann.width / 2) / imgW,
                (ann.y + ann.height / 2) / imgH,
                ann.width / imgW,
                ann.height / imgH,
            ];
            axios.post(`${API_URL}/annotations`, {
                image_id: currentImage.id,
                class_name: className,
                bbox,
            })
                .then(res => {
                    setAnnotations(prev => [...prev, res.data]);
                    pushHistory({ type: 'add', ann: res.data });
                    setImages(prev => prev.map(img =>
                        img.id === currentImage.id ? { ...img, status: 'annotated' } : img
                    ));
                    setAllUsedClasses(prev =>
                        prev.includes(className) ? prev : [...prev, className]
                    );
                    showStatus(`Annotation added: ${className}`);
                })
                .catch(() => setError("Failed to save annotation."));
        }
    };

    const handleClassCancel = () => {
        const wasAI = !!classifyingAnnId;
        setPendingAnnotation(null);
        setPendingPolyline(null);
        setNewAnnotation(null);
        setClassifyingAnnId(null);
        if (wasAI) {
            // Skip this one, move to next in queue
            processNextAIAnnotation();
        }
    };

    // startSeedTraining replaced by TrainingPanel

    const startAutoAnnotation = () => setShowAutoAnnotatePanel(true);

    // Measure the canvas-center container directly to avoid cropping from imprecise offsets
    const measureCanvas = useCallback(() => {
        if (canvasCenterRef.current) {
            const r = canvasCenterRef.current.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
                setCanvasSize({ w: Math.max(r.width, 300), h: Math.max(r.height, 300) });
                return;
            }
        }
        // Fallback: measure the outer area with corrected offsets
        // padding(40) + toolbar(40) + gap(10) + ai-bar(52) + gap(10) = 152 height overhead; 40 width overhead
        if (canvasAreaRef.current) {
            const w = canvasAreaRef.current.clientWidth - 40;
            const h = canvasAreaRef.current.clientHeight - 152;
            setCanvasSize({ w: Math.max(w, 300), h: Math.max(h, 300) });
        }
    }, []);

    useEffect(() => {
        measureCanvas();
        window.addEventListener('resize', measureCanvas);
        return () => window.removeEventListener('resize', measureCanvas);
    }, [measureCanvas]);

    // Re-measure after an image is selected so canvasCenterRef is populated
    useEffect(() => {
        measureCanvas();
    }, [currentImage?.id, measureCanvas]);

    // Reset zoom/pan and selection when switching images
    useEffect(() => {
        setUserZoom(1);
        setStagePos({ x: 0, y: 0 });
        setSelectedAnnId(null);
        setPendingAnnotation(null);
        setNewAnnotation(null);
        setIsDrawing(false);
        setPendingPolyline(null);
        setNewPolylinePoints([]);
        setPolylineCursor(null);
        setHistory([]);
        setRedoStack([]);
        setPreviewAngle(0);
    }, [currentImage?.id]);

    // Sync transformer to selected annotation node
    useEffect(() => {
        const tr = transformerRef.current;
        if (!tr) return;
        const node = selectedAnnId ? annNodesRef.current[selectedAnnId] : null;
        tr.nodes(node ? [node] : []);
        tr.getLayer()?.batchDraw();
    }, [selectedAnnId, annotations]);

    // Keyboard: space=pan, Ctrl+Z=undo, Ctrl+Y / Ctrl+Shift+Z=redo
    useEffect(() => {
        const onKeyDown = (e) => {
            const inTextField = ['INPUT', 'TEXTAREA'].includes(e.target?.tagName) || e.target?.isContentEditable;
            if (e.key === ' ' && !e.repeat) {
                e.preventDefault();
                setIsPanning(true);
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                handleUndo();
            }
            if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
                e.preventDefault();
                handleRedo();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !inTextField && selectedAnnId) {
                e.preventDefault();
                handleCopyAnnotation();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'v' && !inTextField && clipboardAnn) {
                e.preventDefault();
                handlePasteAnnotation();
            }
            if (e.key === 'Escape') {
                setSelectedAnnId(null);
                cancelPolyline();
            }
            if (e.key === 'Enter' && (drawMode === 'polyline' || drawMode === 'segment') && newPolylinePoints.length >= 3) {
                finishPolyline();
            }
            if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnId) {
                handleRejectAnnotation(selectedAnnId);
            }
            // Jump between images — skip while mid-draw or a class picker is open,
            // same guard ReviewPanel uses so arrows don't fight the drawing tools.
            if (
                (e.key === 'ArrowRight' || e.key === 'ArrowLeft') &&
                !inTextField && !pendingAnnotation && !pendingPolyline &&
                newPolylinePoints.length === 0
            ) {
                e.preventDefault();
                handleNavigateImage(e.key === 'ArrowRight' ? 'next' : 'prev');
            }
        };
        const onKeyUp = (e) => {
            if (e.key === ' ') setIsPanning(false);
        };
        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);
        return () => {
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [handleUndo, handleRedo, selectedAnnId, drawMode, newPolylinePoints, finishPolyline, cancelPolyline, clipboardAnn, handleCopyAnnotation, handlePasteAnnotation, pendingAnnotation, pendingPolyline]);

    // The box to draw while mouse is held or while picker is open
    const drawnBox = pendingAnnotation || newAnnotation;
    const drawnPolylinePoints = pendingPolyline || newPolylinePoints;
    const boxAnnotationCount = annotations.filter(a => !['polygon', 'segment'].includes(a.annotation_type)).length;

    return (
        <div className="workspace">
            {/* ── Sidebar ── */}
            <aside className="workspace-sidebar">
                {project.project_type === 'combined' || project.project_type === 'ocr' ? (
                    <div className="sidebar-section sidebar-actions">
                        <p className="sidebar-label">Pipeline (Detect + OCR + Segment)</p>
                        <button className="btn-action" onClick={() => setShowTrainingPanel(true)}>
                            <Rocket size={14} /> Train Seed / Character Detector
                        </button>
                        {seedModelInfo && (
                            <p className="ocr-seed-status">
                                {seedModelInfo.exists
                                    ? '✓ Character detector trained'
                                    : '— Character detector not trained yet'}
                            </p>
                        )}
                        <button className="btn-action btn-action-secondary" onClick={startAutoAnnotation}>
                            <Sparkles size={14} /> Auto-Annotate
                        </button>
                        <button className="btn-action btn-action-al" onClick={() => setShowActiveLearningPanel(true)}>
                            <Brain size={14} /> Active Learning (Detection)
                        </button>
                        <button className="btn-action btn-action-main" onClick={() => setShowMainTrainingPanel(true)}>
                            <Target size={14} /> Train Main Model
                        </button>
                        <button className="btn-action btn-action-ocr" onClick={() => setShowOcrPanel(true)}>
                            <Type size={14} /> Train OCR Model
                        </button>
                        <button
                            className="btn-action btn-action-secondary"
                            onClick={handleOcrAutoLabel}
                            disabled={ocrAutoLabeling || images.filter(img => img.status === 'pending').length === 0}
                            title={`Use the trained ${drawMode === 'segment' ? 'segmentation model' : 'detection model'} to pre-label pending photos as ${drawMode === 'segment' ? 'segmentation masks' : drawMode === 'polyline' ? 'polylines' : 'boxes'} (current canvas tool) — review and correct after`}
                        >
                            <Sparkles size={14} /> {ocrAutoLabeling ? 'Labeling…' : `Auto-Label Characters (${drawMode === 'segment' ? 'Segment' : drawMode === 'polyline' ? 'Polyline' : 'Box'})`}
                        </button>
                        <button
                            className="btn-action btn-action-al"
                            onClick={() => setShowOcrActiveLearningPanel(true)}
                            title="Rank pending photos by how uncertain the trained annotation model is, so you label the hardest ones first"
                        >
                            <Brain size={14} /> Active Learning (OCR)
                        </button>
                        <button
                            className="btn-action btn-action-seg"
                            onClick={() => setShowSegPanel(true)}
                            title="Train an instance-segmentation model on annotations drawn with the Segment tool (mask outlines, not boxes)"
                        >
                            <Scissors size={14} /> Train Segmentation Model
                        </button>
                        <button
                            className="btn-action btn-action-review"
                            onClick={() => setShowReviewPanel(true)}
                            disabled={images.filter(img => img.status === 'annotated').length === 0}
                        >
                            <Eye size={14} /> Review Annotations
                        </button>
                        <button className="btn-action btn-action-labels" onClick={() => setShowLabelsPanel(true)}>
                            <Tag size={14} /> Edit Labels
                        </button>
                        <button className="btn-action btn-action-models" onClick={() => setShowModelsPanel(true)}>
                            <Package size={14} /> View Models
                        </button>
                    </div>
                ) : (
                    <div className="sidebar-section sidebar-actions">
                        <p className="sidebar-label">Pipeline</p>
                        <button className="btn-action" onClick={() => setShowTrainingPanel(true)}>
                            <Rocket size={14} /> Train Seed Model
                        </button>
                        <button className="btn-action btn-action-secondary" onClick={startAutoAnnotation}>
                            <Sparkles size={14} /> Auto-Annotate
                        </button>
                        <button className="btn-action btn-action-al" onClick={() => setShowActiveLearningPanel(true)}>
                            <Brain size={14} /> Active Learning
                        </button>
                        <button
                            className="btn-action btn-action-review"
                            onClick={() => setShowReviewPanel(true)}
                            disabled={images.filter(img => img.status === 'annotated').length === 0}
                        >
                            <Eye size={14} /> Review Annotations
                        </button>
                        <button className="btn-action btn-action-main" onClick={() => setShowMainTrainingPanel(true)}>
                            <Target size={14} /> Train Main Model
                        </button>
                        <button className="btn-action btn-action-labels" onClick={() => setShowLabelsPanel(true)}>
                            <Tag size={14} /> Edit Labels
                        </button>
                        <button className="btn-action btn-action-models" onClick={() => setShowModelsPanel(true)}>
                            <Package size={14} /> View Models
                        </button>
                    </div>
                )}

                <div className="sidebar-section">
                    <p className="sidebar-label">Images ({images.length})</p>
                    <button className="btn-action btn-action-video" onClick={() => setShowVideoPanel(true)}>
                        <Film size={14} /> Import Video
                    </button>
                    <button
                        className="btn-action btn-action-sequence"
                        onClick={() => setShowSequencePanel(true)}
                        title="Define ordered region checkpoints an object must pass through, in order (e.g. bulb visiting socket 1 → 2 → 3)"
                    >
                        <Target size={14} /> Sequence Detection
                    </button>
                    <label className={`upload-btn ${uploading ? 'uploading' : ''}`}>
                        {uploading
                            ? `Uploading ${uploadFileCount} file${uploadFileCount !== 1 ? 's' : ''}…`
                            : '+ Upload Images'}
                        <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            accept="image/*"
                            style={{ display: 'none' }}
                            onChange={handleFileUpload}
                            disabled={uploading}
                        />
                    </label>

                    {/* ── Dataset export/import — download this project's images+labels
                        as a zip, or import one exported from another project ── */}
                    <button
                        className="btn-action btn-action-secondary"
                        onClick={handleExportDataset}
                        disabled={exportingDataset || images.length === 0}
                        title="Download all images + annotations (boxes, polylines, segment masks) as a zip"
                    >
                        {exportingDataset ? `Downloading… ${exportProgress || ''}` : '⬇ Download Dataset'}
                    </button>
                    <label className={`upload-btn ${importingDataset ? 'uploading' : ''}`} title="Import a dataset zip exported from another project — images + all annotation types are recreated here">
                        {importingDataset ? 'Importing…' : '⬆ Import Dataset (.zip)'}
                        <input
                            ref={datasetImportRef}
                            type="file"
                            accept=".zip"
                            style={{ display: 'none' }}
                            onChange={handleImportDataset}
                            disabled={importingDataset}
                        />
                    </label>

                    {/* Progress bar — visible only while uploading */}
                    {uploading && (
                        <div className="upload-progress-wrap">
                            <div className="upload-progress-bar">
                                <div
                                    className="upload-progress-fill"
                                    style={{ width: `${uploadProgress ?? 0}%` }}
                                />
                            </div>
                            <span className="upload-progress-pct">
                                {uploadProgress ?? 0}%
                            </span>
                        </div>
                    )}
                </div>

                {/* Image list — also the drag & drop target */}
                <div
                    className={`image-list ${isDragOver ? 'drag-over' : ''}`}
                    onDragOver={handleDragOver}
                    onDragEnter={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    {images.length === 0 ? (
                        <div className="sidebar-empty-drop">
                            <p className="sidebar-empty">No images yet.</p>
                            <p className="drop-hint">Upload or drop images here</p>
                        </div>
                    ) : (
                        <>
                            {/* AL suggestion filter banner */}
                            {suggestedImageIds && (
                                <div className="al-filter-banner">
                                    <span><Target size={11} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />{suggestedImageIds.size} suggested</span>
                                    <button className="al-filter-clear" onClick={() => setSuggestedImageIds(null)}>Clear</button>
                                </div>
                            )}
                            {images.map(img => {
                                const isSuggested = suggestedImageIds?.has(String(img.id));
                                return (
                                <div
                                    key={img.id}
                                    className={`image-item ${currentImage?.id === img.id ? 'active' : ''} ${isSuggested ? 'image-item--suggested' : ''}`}
                                    onClick={() => handleImageClick(img)}
                                >
                                    <span className="image-item-dot"></span>
                                    <span className="image-item-name">{img.filename}</span>
                                    {isSuggested && <span className="image-item-al-badge"><Target size={9} /></span>}
                                    <span className={`image-item-status status-${img.status}`}>
                                        {img.status}
                                    </span>
                                </div>
                                );
                            })}
                            {/* Drop overlay shown on top of the list while dragging */}
                            {isDragOver && (
                                <div className="drop-overlay">
                                    <span className="drop-overlay-icon"><Upload size={24} /></span>
                                    <span>Drop to add images</span>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </aside>

            {/* ── Canvas Area ── */}
            <div className="workspace-canvas-area" ref={canvasAreaRef}>
                {error && (
                    <div className="error-banner">
                        <span className="error-icon"><AlertTriangle size={16} /></span>
                        {error}
                        <button className="error-dismiss" onClick={() => setError(null)}><X size={16} /></button>
                    </div>
                )}

                {statusMsg && <div className="status-toast">{statusMsg}</div>}

                {currentImage ? (
                    <div className="canvas-wrapper">
                        <div className="canvas-toolbar">
                            <div className="canvas-nav">
                                <button
                                    className="btn-toolbar"
                                    onClick={() => handleNavigateImage('prev')}
                                    disabled={images.findIndex(img => img.id === currentImage.id) <= 0}
                                    title="Previous image (←)"
                                ><ArrowLeft size={14} /></button>
                                <span className="canvas-nav-count">
                                    {images.findIndex(img => img.id === currentImage.id) + 1} / {images.length}
                                </span>
                                <button
                                    className="btn-toolbar"
                                    onClick={() => handleNavigateImage('next')}
                                    disabled={images.findIndex(img => img.id === currentImage.id) >= images.length - 1}
                                    title="Next image (→)"
                                ><ArrowRight size={14} /></button>
                            </div>
                            <span className="canvas-filename">{currentImage.filename}</span>
                            <span className="canvas-dims">{imgW} × {imgH}px</span>
                            <span
                                className="canvas-hint-icon"
                                title={
                                    (isPanning
                                        ? 'Pan mode — drag to move'
                                        : selectedAnnId
                                            ? 'Drag to move · handles to resize · double-click to fix the label'
                                            : (drawMode === 'polyline' || drawMode === 'segment')
                                                ? 'Click to place points · double-click or Enter to finish · Esc to cancel · double-click a shape to fix its label'
                                                : 'Draw a box to annotate · double-click a shape to fix its label')
                                    + ' · Scroll to zoom · hold Space + drag to pan'
                                }
                            ><Info size={14} /></span>
                            {/* ── Draw mode: Box vs Polyline vs Segment ── */}
                            <div className="draw-mode-toggle">
                                <button
                                    className={`btn-toolbar ${drawMode === 'box' ? 'btn-toolbar--active' : ''}`}
                                    onClick={() => changeDrawMode('box')}
                                    title="Box tool — axis-aligned rectangle"
                                ><Square size={14} /></button>
                                <button
                                    className={`btn-toolbar ${drawMode === 'polyline' ? 'btn-toolbar--active' : ''}`}
                                    onClick={() => changeDrawMode('polyline')}
                                    title="Polyline tool — trace the character's true outline. Best for angled LHS/RHS plate views where boxes would touch."
                                ><PenTool size={14} /></button>
                                {(project.project_type === 'combined' || project.project_type === 'ocr') && (
                                    <button
                                        className={`btn-toolbar ${drawMode === 'segment' ? 'btn-toolbar--active' : ''}`}
                                        onClick={() => changeDrawMode('segment')}
                                        title="Segment tool — trace the real mask outline for instance-segmentation training (separate from the polyline precision tool)"
                                    ><Scissors size={14} /></button>
                                )}
                            </div>
                            <button
                                className="btn-toolbar btn-toolbar--labeled"
                                onClick={handleConvertToPolyline}
                                disabled={boxAnnotationCount === 0}
                                title="Replace this image's box annotations with adjustable polylines, so you can drag each corner onto the real (rotated) character outline"
                            ><RefreshCw size={14} /> Box → Polyline</button>
                            {/* ── Undo / Redo ── */}
                            <button className="btn-toolbar" onClick={handleUndo} disabled={!history.length} title="Undo (Ctrl+Z)"><Undo2 size={14} /></button>
                            <button className="btn-toolbar" onClick={handleRedo} disabled={!redoStack.length} title="Redo (Ctrl+Y)"><Redo2 size={14} /></button>
                            {/* ── Copy / paste a shape (box, polygon or segment) ── */}
                            <button className="btn-toolbar" onClick={handleCopyAnnotation} disabled={!selectedAnnId} title="Copy selected shape (Ctrl+C)"><Copy size={14} /></button>
                            <button className="btn-toolbar" onClick={handlePasteAnnotation} disabled={!clipboardAnn} title="Paste copied shape (Ctrl+V) — then drag it into place"><ClipboardPaste size={14} /></button>
                            {/* ── Clear all annotations ── */}
                            <button className="btn-toolbar" onClick={handleClearAllAnnotations} disabled={annotations.length === 0} title="Delete all annotations on this image"><Trash2 size={14} /></button>
                            {/* ── Delete image ── */}
                            <button className="btn-toolbar btn-toolbar--danger" onClick={handleDeleteImage} title="Delete this image (and all its annotations)"><ImageOff size={14} /></button>
                            {/* ── Rotate (portrait ↔ landscape; boxes are remapped) ── */}
                            <button className="btn-toolbar" onClick={() => handleRotateImage('ccw')} title="Rotate 90° counter-clockwise"><RotateCcw size={14} /></button>
                            <button className="btn-toolbar" onClick={() => handleRotateImage('cw')} title="Rotate 90° clockwise"><RotateCw size={14} /></button>
                            {/* ── OCR/combined: smart rotate + fine nudge + alignment grid ── */}
                            {(project.project_type === 'ocr' || project.project_type === 'combined') && (<>
                            <button
                                className={`btn-toolbar btn-toolbar--smart ${isStraightening ? 'busy' : ''}`}
                                onClick={handleAutoStraighten}
                                disabled={isStraightening}
                                title="Smart rotate — auto-detect the tilt of the engraved text and level the image"
                            ><Wand2 size={14} /></button>
                            <div className="rotate-slider-group" title="Free rotate — drag the slider (or type an exact angle) to rotate live against the grid, then Apply to save">
                                <input
                                    type="range"
                                    className="rotate-slider"
                                    min="-180" max="180" step="0.2"
                                    value={previewAngle}
                                    onChange={e => setPreviewAngle(parseFloat(e.target.value))}
                                    onDoubleClick={() => setPreviewAngle(0)}
                                />
                                <input
                                    type="number"
                                    className="rotate-angle-input"
                                    min="-180" max="180" step="0.1"
                                    value={previewAngle}
                                    onChange={e => {
                                        const v = parseFloat(e.target.value);
                                        if (!Number.isNaN(v)) setPreviewAngle(Math.max(-180, Math.min(180, v)));
                                    }}
                                />
                                <span className="fine-rotate-unit">°</span>
                                {previewAngle !== 0 && (<>
                                    <button className="btn-toolbar btn-toolbar--apply" onClick={handleApplyPreviewRotation} title="Bake this rotation into the image">Apply</button>
                                    <button className="btn-toolbar" onClick={() => setPreviewAngle(0)} title="Discard preview rotation"><X size={12} /></button>
                                </>)}
                            </div>
                            {/* ── Grid overlay toggle + spacing ── */}
                            <button
                                className={`btn-toolbar ${showGrid ? 'btn-toolbar--active' : ''}`}
                                onClick={() => setShowGrid(g => !g)}
                                title="Toggle alignment grid — level the plate against the lines before boxing characters"
                            ><Grid3x3 size={14} /></button>
                            {showGrid && (
                                <select
                                    className="grid-spacing-select"
                                    value={gridSpacing}
                                    onChange={e => setGridSpacing(Number(e.target.value))}
                                    title="Grid spacing (image pixels)"
                                >
                                    <option value={20}>20px</option>
                                    <option value={50}>50px</option>
                                    <option value={100}>100px</option>
                                    <option value={200}>200px</option>
                                </select>
                            )}
                            </>)}
                            {/* ── Zoom ── */}
                            <button className="btn-toolbar" onClick={() => { setUserZoom(z => Math.min(15, z * 1.3)); }} title="Zoom in"><ZoomIn size={14} /></button>
                            <span style={{ fontSize: 11, color: '#666', minWidth: 36, textAlign: 'center' }}>{Math.round(userZoom * 100)}%</span>
                            <button className="btn-toolbar" onClick={() => { setUserZoom(z => Math.max(0.2, z / 1.3)); }} title="Zoom out"><ZoomOut size={14} /></button>
                            <button className="btn-toolbar" onClick={resetZoom} title="Reset view"><Maximize2 size={14} /></button>
                            <span className="canvas-ann-count">
                                {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
                            </span>
                            {annotations.length === 0 && (
                                <button
                                    className="btn-no-objects"
                                    onClick={handleMarkEmpty}
                                    title="No objects in this frame — mark as done and advance to next image"
                                >
                                    <Check size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} /> No Objects
                                </button>
                            )}
                        </div>

                        {/* ── AI Prompt Bar (Below Toolbar) ── */}
                        <div className="ai-prompt-bar">
                            <div className="ai-prompt-bar-left">
                                <span className="ai-prompt-spark"><Sparkles size={16} /></span>
                                <input
                                    type="text"
                                    className="ai-prompt-bar-input"
                                    placeholder="Enter object to detect (e.g. 'hard hat')..."
                                    value={aiPrompt}
                                    onChange={(e) => setAiPrompt(e.target.value)}
                                    disabled={isDetecting}
                                />
                                <label className="ai-prompt-checkbox" title="Clear current annotations before running AI detection">
                                    <input
                                        type="checkbox"
                                        checked={clearExisting}
                                        onChange={e => setClearExisting(e.target.checked)}
                                    />
                                    <span>Clear before detect</span>
                                </label>
                            </div>
                            <div className="ai-prompt-bar-right">
                                <button
                                    className={`ai-bar-btn ai-bar-btn-primary ${isDetecting ? 'loading' : ''}`}
                                    onClick={handleAIDetect}
                                    disabled={isDetecting || !aiPrompt.trim()}
                                >
                                    {isDetecting ? 'Detecting...' : 'Detect Image'}
                                </button>
                                <button
                                    className={`ai-bar-btn ai-bar-btn-secondary ${isDetecting ? 'loading' : ''}`}
                                    onClick={handleAIApplyAll}
                                    disabled={isDetecting || !aiPrompt.trim()}
                                >
                                    Apply to All
                                </button>
                            </div>
                        </div>

                        <div className="canvas-center" ref={canvasCenterRef} style={{ overflow: 'hidden', cursor: isPanning ? 'grab' : 'crosshair' }}>
                            <Stage
                                className="canvas-stage"
                                width={canvasSize.w}
                                height={canvasSize.h}
                                scaleX={scale * userZoom}
                                scaleY={scale * userZoom}
                                x={Math.round((canvasSize.w - stageW) / 2) + stagePos.x}
                                y={Math.round((canvasSize.h - stageH) / 2) + stagePos.y}
                                draggable={isPanning}
                                onWheel={handleWheel}
                                onDragEnd={(e) => {
                                    const cx = Math.round((canvasSize.w - stageW) / 2);
                                    const cy = Math.round((canvasSize.h - stageH) / 2);
                                    setStagePos({ x: e.target.x() - cx, y: e.target.y() - cy });
                                }}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onDblClick={(drawMode === 'polyline' || drawMode === 'segment') ? finishPolyline : undefined}
                            >
                                <Layer>
                                    {/* Rotates live with the slider; grid stays fixed as the level reference */}
                                    <Group
                                        x={imgW / 2} y={imgH / 2}
                                        offsetX={imgW / 2} offsetY={imgH / 2}
                                        rotation={previewAngle}
                                    >
                                    <KonvaImage
                                        src={`${API_URL.replace("/api/v1", "")}${currentImage.filepath}${imgVersion[currentImage.id] ? `?v=${imgVersion[currentImage.id]}` : ''}`}
                                        onLoad={handleImageLoad}
                                        listening={false}
                                    />
                                    {annotations.map(ann => {
                                        const bx = (ann.bbox[0] - ann.bbox[2] / 2) * imgW;
                                        const by = (ann.bbox[1] - ann.bbox[3] / 2) * imgH;
                                        const bw = ann.bbox[2] * imgW;
                                        const bh = ann.bbox[3] * imgH;
                                        const unclassified = ann.source === 'ai_prompt' && !ann.class_name;
                                        const isSegment = ann.annotation_type === 'segment';
                                        const color = unclassified ? '#f59e0b' : ann.source === 'auto' ? '#a78bfa' : isSegment ? '#10b981' : '#f43f5e';
                                        const isPolyShape = (ann.annotation_type === 'polygon' || isSegment) && ann.points;
                                        const isSelected = selectedAnnId === ann.id;
                                        const totalScale = scale * userZoom;
                                        const fontSize = Math.max(10, 13 / totalScale);
                                        const padX = 4 / totalScale;
                                        const padY = 2 / totalScale;
                                        const labelH = fontSize + padY * 2;
                                        return (
                                            <React.Fragment key={ann.id}>
                                                {isPolyShape ? (
                                                    <PolygonAnnotation
                                                        ann={ann}
                                                        imgW={imgW} imgH={imgH}
                                                        color={color}
                                                        isSelected={isSelected}
                                                        isPanning={isPanning}
                                                        totalScale={totalScale}
                                                        onSelect={(e) => handleAnnClick(ann.id, e)}
                                                        onRelabel={(e) => handleEditLabel(ann, e)}
                                                        onMoveEnd={(pts) => handlePolygonMoveEnd(ann, pts)}
                                                        onVertexDragEnd={(pts) => handlePolygonVertexEnd(ann, pts)}
                                                        isDraggingShapeRef={isDraggingShapeRef}
                                                    />
                                                ) : (
                                                    <Rect
                                                        ref={node => { if (node) annNodesRef.current[ann.id] = node; else delete annNodesRef.current[ann.id]; }}
                                                        x={bx} y={by} width={bw} height={bh}
                                                        stroke={isSelected ? '#facc15' : color}
                                                        strokeWidth={(isSelected ? 2 : 1.5) / totalScale}
                                                        fill={unclassified
                                                            ? 'rgba(245,158,11,0.10)'
                                                            : ann.source === 'auto'
                                                                ? 'rgba(167,139,250,0.07)'
                                                                : 'rgba(244,63,94,0.07)'}
                                                        dash={unclassified ? [6 / totalScale, 3 / totalScale] : undefined}
                                                        draggable={isSelected && !isPanning}
                                                        onClick={(e) => handleAnnClick(ann.id, e)}
                                                        onTap={(e) => handleAnnClick(ann.id, e)}
                                                        onDblClick={(e) => handleEditLabel(ann, e)}
                                                        onDblTap={(e) => handleEditLabel(ann, e)}
                                                        onDragStart={() => { isDraggingShapeRef.current = true; }}
                                                        onDragEnd={(e) => { isDraggingShapeRef.current = false; handleAnnDragEnd(e, ann); }}
                                                        onTransformEnd={(e) => handleAnnTransformEnd(e, ann)}
                                                        onMouseEnter={e => { e.target.getStage().container().style.cursor = 'move'; }}
                                                        onMouseLeave={e => { e.target.getStage().container().style.cursor = isPanning ? 'grab' : 'crosshair'; }}
                                                    />
                                                )}
                                                {/* Label background */}
                                                <Rect
                                                    x={bx} y={by - labelH}
                                                    width={Math.min(bw, ((unclassified ? 12 : ann.class_name.length) * fontSize * 0.6) + padX * 2 + 30/totalScale)}
                                                    height={labelH}
                                                    fill={isSelected ? '#facc15' : color}
                                                    cornerRadius={2 / totalScale}
                                                    listening={false}
                                                />
                                                <Text
                                                    x={bx + padX}
                                                    y={by - labelH + padY}
                                                    text={unclassified ? '? unclassified' : `${ann.class_name}${ann.source === 'auto' ? ' ✦' : ''}`}
                                                    fontSize={fontSize}
                                                    fontFamily="sans-serif"
                                                    fill="#fff"
                                                    fontStyle="bold"
                                                    listening={false}
                                                />

                                                {/* ── Canvas Controls (Only for AI boxes) ── */}
                                                {ann.source !== 'manual' && (
                                                    <Group x={bx + bw - (44 / totalScale)} y={by + (4 / totalScale)}>
                                                        <Group
                                                            onClick={(e) => { e.cancelBubble = true; handleRejectAnnotation(ann.id); }}
                                                            onTap={(e) => { e.cancelBubble = true; handleRejectAnnotation(ann.id); }}
                                                            onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer'; }}
                                                            onMouseLeave={e => { e.target.getStage().container().style.cursor = isPanning ? 'grab' : 'crosshair'; }}
                                                        >
                                                            <Rect width={18 / totalScale} height={18 / totalScale} fill="#f43f5e" cornerRadius={3 / totalScale} shadowBlur={2 / totalScale} />
                                                            <Text text="✕" fill="#fff" fontSize={12 / totalScale} x={5 / totalScale} y={3 / totalScale} fontStyle="bold" />
                                                        </Group>
                                                        <Group
                                                            x={22 / totalScale}
                                                            onClick={(e) => { e.cancelBubble = true; handleAcceptAnnotation(ann.id); }}
                                                            onTap={(e) => { e.cancelBubble = true; handleAcceptAnnotation(ann.id); }}
                                                            onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer'; }}
                                                            onMouseLeave={e => { e.target.getStage().container().style.cursor = isPanning ? 'grab' : 'crosshair'; }}
                                                        >
                                                            <Rect width={18 / totalScale} height={18 / totalScale} fill="#22c55e" cornerRadius={3 / totalScale} shadowBlur={2 / totalScale} />
                                                            <Text text="✓" fill="#fff" fontSize={12 / totalScale} x={4 / totalScale} y={3 / totalScale} fontStyle="bold" />
                                                        </Group>
                                                    </Group>
                                                )}
                                            </React.Fragment>
                                        );
                                    })}
                                    {drawnBox && (
                                        <Rect
                                            x={drawnBox.x}
                                            y={drawnBox.y}
                                            width={drawnBox.width}
                                            height={drawnBox.height}
                                            stroke="#dc143c"
                                            strokeWidth={2 / (scale * userZoom)}
                                            dash={[6 / (scale * userZoom), 3 / (scale * userZoom)]}
                                            fill="rgba(220,20,60,0.08)"
                                        />
                                    )}
                                    {drawnPolylinePoints.length > 0 && (() => {
                                        const ts = scale * userZoom;
                                        const flat = drawnPolylinePoints.flatMap(p => [p.x, p.y]);
                                        const withCursor = (!pendingPolyline && polylineCursor)
                                            ? [...flat, polylineCursor.x, polylineCursor.y]
                                            : flat;
                                        return (
                                            <>
                                                <Line
                                                    points={withCursor}
                                                    closed={!!pendingPolyline}
                                                    stroke="#dc143c"
                                                    strokeWidth={2 / ts}
                                                    dash={[6 / ts, 3 / ts]}
                                                    fill={pendingPolyline ? 'rgba(220,20,60,0.08)' : undefined}
                                                    listening={false}
                                                />
                                                {drawnPolylinePoints.map((p, i) => (
                                                    <Circle
                                                        key={i}
                                                        x={p.x} y={p.y}
                                                        radius={4 / ts}
                                                        fill={i === 0 ? '#facc15' : '#dc143c'}
                                                        stroke="#fff"
                                                        strokeWidth={1 / ts}
                                                        listening={false}
                                                    />
                                                ))}
                                            </>
                                        );
                                    })()}
                                    </Group>
                                    {/* ── Alignment grid overlay — fixed, drawn above the (possibly rotating) image ── */}
                                    {showGrid && (() => {
                                        const ts = scale * userZoom;
                                        const lines = [];
                                        for (let x = gridSpacing, i = 1; x < imgW; x += gridSpacing, i++) {
                                            lines.push(
                                                <Line key={`gv${x}`} points={[x, 0, x, imgH]}
                                                    stroke={i % 5 === 0 ? 'rgba(34,211,238,0.55)' : 'rgba(34,211,238,0.25)'}
                                                    strokeWidth={(i % 5 === 0 ? 1.4 : 0.8) / ts}
                                                    listening={false} />
                                            );
                                        }
                                        for (let y = gridSpacing, i = 1; y < imgH; y += gridSpacing, i++) {
                                            lines.push(
                                                <Line key={`gh${y}`} points={[0, y, imgW, y]}
                                                    stroke={i % 5 === 0 ? 'rgba(34,211,238,0.55)' : 'rgba(34,211,238,0.25)'}
                                                    strokeWidth={(i % 5 === 0 ? 1.4 : 0.8) / ts}
                                                    listening={false} />
                                            );
                                        }
                                        return lines;
                                    })()}
                                    <Transformer
                                        ref={transformerRef}
                                        rotateEnabled={false}
                                        boundBoxFunc={(oldBox, newBox) =>
                                            newBox.width < 5 || newBox.height < 5 ? oldBox : newBox
                                        }
                                    />
                                </Layer>
                            </Stage>

                            {/* Class picker floats over the canvas */}
                            {(pendingAnnotation || pendingPolyline) && (
                                <ClassPicker
                                    classes={localClasses}
                                    usedClasses={allUsedClasses}
                                    onConfirm={handleClassConfirm}
                                    onCancel={handleClassCancel}
                                    remaining={classifyingAnnId ? aiQueueRef.current.length + 1 : 0}
                                    ocrMode={project.project_type === 'ocr' || project.project_type === 'combined'}
                                />
                            )}
                        </div>

                        {annotations.length > 0 && (
                            <div className="annotations-list">
                                <div className="annotations-list-header">
                                    <p className="sidebar-label" style={{ margin: 0 }}>Annotations</p>
                                    {annotations.some(a => a.source !== 'manual') && (
                                        <button className="accept-all-btn" onClick={handleAcceptAll}>
                                            Accept All
                                        </button>
                                    )}
                                </div>
                                {annotations.map(ann => (
                                    <div key={ann.id} className="annotation-row">
                                        <span className="annotation-dot" style={{ backgroundColor: ann.source === 'auto' ? '#a78bfa' : '#f43f5e' }}></span>
                                        <span
                                            className="annotation-class annotation-class--editable"
                                            title="Tap to fix this label"
                                            onClick={() => handleEditLabel(ann)}
                                        >{ann.class_name}</span>
                                        <span className={`annotation-source source-${ann.source}`}>
                                            {ann.source}
                                        </span>
                                        {ann.source !== 'manual' && (
                                            <div className="annotation-verify-actions">
                                                <button
                                                    className="ann-btn ann-btn-accept"
                                                    title="Accept"
                                                    onClick={() => handleAcceptAnnotation(ann.id)}
                                                ><Check size={10} /></button>
                                                <button
                                                    className="ann-btn ann-btn-reject"
                                                    title="Reject"
                                                    onClick={() => handleRejectAnnotation(ann.id)}
                                                ><X size={10} /></button>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="canvas-placeholder">
                        <div className="placeholder-icon"><ImageIcon size={48} /></div>
                        <h3>Select an image</h3>
                        <p>Choose an image from the sidebar to start annotating.</p>
                    </div>
                )}
            </div>
            {showTrainingPanel && (
                <TrainingPanel
                    project={project}
                    onClose={() => {
                        setShowTrainingPanel(false);
                        if (project.project_type === 'ocr' || project.project_type === 'combined') loadSeedModelInfo();
                    }}
                />
            )}
            {showAutoAnnotatePanel && (
                <AutoAnnotatePanel
                    project={project}
                    onClose={() => setShowAutoAnnotatePanel(false)}
                    onAnnotationsUpdated={() => {
                        // Refresh image list so status changes are reflected
                        axios.get(`${API_URL}/images/project/${project.id}`)
                            .then(res => setImages(res.data))
                            .catch(() => {});
                    }}
                />
            )}
            {showMainTrainingPanel && (
                <MainTrainingPanel
                    project={project}
                    onClose={() => setShowMainTrainingPanel(false)}
                />
            )}
            {showLabelsPanel && (
                <LabelsPanel
                    project={{ ...project, classes: localClasses }}
                    onClose={() => setShowLabelsPanel(false)}
                    onEditClassImages={handleEditClassImages}
                    onLabelsUpdated={(updatedClasses) => {
                        setLocalClasses(updatedClasses);
                        // Remove any used-class entries that were deleted or renamed
                        setAllUsedClasses(prev =>
                            prev.filter(c => updatedClasses.includes(c))
                        );
                        // Propagate to parent (App.js) so project card stays in sync
                        if (onProjectUpdated) {
                            onProjectUpdated({ ...project, classes: updatedClasses });
                        }
                    }}
                />
            )}
            {showReviewPanel && (
                <ReviewPanel
                    project={{ ...project, classes: localClasses }}
                    images={images}
                    filterImageIds={reviewFilterIds}
                    filterLabel={reviewFilterLabel}
                    filterClassName={reviewFilterClassName}
                    onClose={() => {
                        setShowReviewPanel(false);
                        setReviewFilterIds(null);
                        setReviewFilterLabel(null);
                        setReviewFilterClassName(null);
                    }}
                    onAnnotationsUpdated={() => {
                        // Refresh images and reload current image annotations after review
                        axios.get(`${API_URL}/images/project/${project.id}`)
                            .then(res => setImages(res.data))
                            .catch(() => {});
                        if (currentImage) {
                            axios.get(`${API_URL}/annotations/image/${currentImage.id}`)
                                .then(res => setAnnotations(res.data))
                                .catch(() => {});
                        }
                    }}
                />
            )}
            {showModelsPanel && (
                <ModelsPanel
                    project={project}
                    onClose={() => setShowModelsPanel(false)}
                    onGoToTrain={(type) => {
                        if (type === 'seed') setShowTrainingPanel(true);
                        else if (type === 'main') setShowMainTrainingPanel(true);
                        else setShowSegPanel(true); // 'seg_seed' | 'seg_main'
                    }}
                />
            )}
            {showOcrPanel && (
                <OcrTrainingPanel
                    project={project}
                    onClose={() => setShowOcrPanel(false)}
                />
            )}
            {showSegPanel && (
                <SegTrainingPanel
                    project={project}
                    onClose={() => setShowSegPanel(false)}
                />
            )}
            {showVideoPanel && (
                <VideoPanel
                    project={project}
                    onClose={() => setShowVideoPanel(false)}
                    onFramesExtracted={() => {
                        // Refresh image list so extracted frames appear
                        axios.get(`${API_URL}/images/project/${project.id}`)
                            .then(res => setImages(res.data))
                            .catch(() => {});
                    }}
                />
            )}
            {showSequencePanel && (
                <SequencePanel
                    project={project}
                    onClose={() => setShowSequencePanel(false)}
                />
            )}
            {showActiveLearningPanel && (
                <ActiveLearningPanel
                    project={project}
                    onClose={() => setShowActiveLearningPanel(false)}
                    onAnnotateImages={handleAnnotateImages}
                    onAnnotationsUpdated={() => {
                        // Refresh image list so curriculum-annotated images are reflected
                        axios.get(`${API_URL}/images/project/${project.id}`)
                            .then(res => setImages(res.data))
                            .catch(() => {});
                    }}
                />
            )}
            {showOcrActiveLearningPanel && (
                <OcrActiveLearningPanel
                    project={project}
                    drawMode={drawMode}
                    onClose={() => setShowOcrActiveLearningPanel(false)}
                    onAnnotateImages={handleAnnotateImages}
                />
            )}
        </div>
    );
};

export default AnnotationWorkspace;
