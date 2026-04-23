import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Stage, Layer, Rect, Text, Image, Group } from 'react-konva';
import useImage from 'use-image';
import TrainingPanel from './TrainingPanel';
import AutoAnnotatePanel from './AutoAnnotatePanel';
import MainTrainingPanel from './MainTrainingPanel';
import LabelsPanel from './LabelsPanel';
import ModelsPanel from './ModelsPanel';
import ReviewPanel from './ReviewPanel';
import VideoPanel from './VideoPanel';
import ActiveLearningPanel from './ActiveLearningPanel';
import './AnnotationWorkspace.css';
import { Sparkles, AlertTriangle, X, Upload, Image as ImageIcon, Check, ArrowLeft, ArrowRight, Brain, Rocket, Eye, Target, Tag, Package, Film } from 'lucide-react';

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

// ── Class Picker ────────────────────────────────────────────────
const ClassPicker = ({ classes, usedClasses, onConfirm, onCancel, remaining = 0 }) => {
    const [customClass, setCustomClass] = useState('');
    const inputRef = useRef(null);

    // Merge project classes + already-used classes, dedupe, keep order
    const presetSet = new Set(classes);
    const usedOnly = usedClasses.filter(c => !presetSet.has(c));
    const allOptions = [...classes, ...usedOnly]; // presets first, then extra used ones

    useEffect(() => {
        if (allOptions.length === 0 && inputRef.current) {
            inputRef.current.focus();
        }
    }, []); // eslint-disable-line

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
    const [isDrawing, setIsDrawing] = useState(false);
    const [newAnnotation, setNewAnnotation] = useState(null);
    const [pendingAnnotation, setPendingAnnotation] = useState(null); // bbox waiting for class
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
    const [showActiveLearningPanel, setShowActiveLearningPanel] = useState(false);
    const [suggestedImageIds, setSuggestedImageIds] = useState(null);  // Set<id> or null (sidebar highlight)
    const [reviewFilterIds, setReviewFilterIds] = useState(null);      // Set<id> or null (ReviewPanel filter)
    // Local copy of classes so edits from LabelsPanel are reflected instantly
    const [localClasses, setLocalClasses] = useState(project.classes || []);
    const [aiPrompt, setAiPrompt] = useState('');
    const [clearExisting, setClearExisting] = useState(false);
    const [isDetecting, setIsDetecting] = useState(false);
    const [classifyingAnnId, setClassifyingAnnId] = useState(null); // id of AI ann being classified
    const aiQueueRef = useRef([]); // queue of {id, bbox} waiting for ClassPicker
    const canvasAreaRef = useRef(null);
    const canvasCenterRef = useRef(null);

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
        try {
            await axios.delete(`${API_URL}/annotations/${annId}`);
            setAnnotations(prev => prev.filter(a => a.id !== annId));
        } catch (e) {
            setError("Failed to delete annotation.");
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

    // Called from ActiveLearningPanel when user clicks "Annotate These"
    const handleAnnotateImages = (imageIds) => {
        const idSet = new Set(imageIds.map(String));
        setSuggestedImageIds(idSet);   // highlight in sidebar
        setReviewFilterIds(idSet);     // open in ReviewPanel
        setShowReviewPanel(true);
    };

    const handleImageLoad = useCallback((w, h) => {
        setLoadedImageSize({ width: w, height: h });
    }, []);

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
                setImages(prev => [...prev, ...res.data]);
                showStatus(`✓ ${res.data.length} image${res.data.length !== 1 ? 's' : ''} uploaded`);
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
        if (pendingAnnotation) return; // class picker is open
        const pos = e.target.getStage().getPointerPosition();
        const x = pos.x / scale;
        const y = pos.y / scale;
        setIsDrawing(true);
        setNewAnnotation({ x, y, width: 0, height: 0 });
    };

    const handleMouseMove = (e) => {
        if (!isDrawing) return;
        const pos = e.target.getStage().getPointerPosition();
        const x = pos.x / scale;
        const y = pos.y / scale;
        setNewAnnotation(prev => ({
            ...prev,
            width: x - prev.x,
            height: y - prev.y,
        }));
    };

    const handleMouseUp = () => {
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

    const handleClassConfirm = (className) => {
        const ann = pendingAnnotation;
        const annId = classifyingAnnId;
        setPendingAnnotation(null);
        setNewAnnotation(null);
        setClassifyingAnnId(null);

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

    // Use the browser-reported natural dimensions once the image loads (EXIF-corrected).
    // Fall back to the backend-stored values while the image is still loading.
    const imgW = loadedImageSize?.width  || currentImage?.width  || 1;
    const imgH = loadedImageSize?.height || currentImage?.height || 1;

    // Fit image inside available canvas area, preserving aspect ratio
    const scale = currentImage
        ? Math.min(1, canvasSize.w / imgW, canvasSize.h / imgH)
        : 1;
    const stageW = currentImage ? Math.round(imgW * scale) : canvasSize.w;
    const stageH = currentImage ? Math.round(imgH * scale) : canvasSize.h;

    // The box to draw while mouse is held or while picker is open
    const drawnBox = pendingAnnotation || newAnnotation;

    return (
        <div className="workspace">
            {/* ── Sidebar ── */}
            <aside className="workspace-sidebar">
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

                <div className="sidebar-section">
                    <p className="sidebar-label">Images ({images.length})</p>
                    <button className="btn-action btn-action-video" onClick={() => setShowVideoPanel(true)}>
                        <Film size={14} /> Import Video
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
                            <span className="canvas-filename">{currentImage.filename}</span>
                            <span className="canvas-dims">{imgW} × {imgH}px</span>
                            <span className="canvas-hint">Draw a box to annotate</span>
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
                                <label className="ai-prompt-checkbox">
                                    <input 
                                        type="checkbox" 
                                        checked={clearExisting} 
                                        onChange={e => setClearExisting(e.target.checked)} 
                                    />
                                    <span>Wipe Existing</span>
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

                        <div className="canvas-center" ref={canvasCenterRef}>
                            <Stage
                                className="canvas-stage"
                                width={stageW}
                                height={stageH}
                                scaleX={scale}
                                scaleY={scale}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                            >
                                <Layer>
                                    <KonvaImage
                                        src={`${API_URL.replace("/api/v1", "")}${currentImage.filepath}`}
                                        onLoad={handleImageLoad}
                                    />
                                    {annotations.map(ann => {
                                        const bx = (ann.bbox[0] - ann.bbox[2] / 2) * imgW;
                                        const by = (ann.bbox[1] - ann.bbox[3] / 2) * imgH;
                                        const bw = ann.bbox[2] * imgW;
                                        const bh = ann.bbox[3] * imgH;
                                        const unclassified = ann.source === 'ai_prompt' && !ann.class_name;
                                        // manual = rose red, auto = violet, unclassified = amber
                                        const color = unclassified ? '#f59e0b' : ann.source === 'auto' ? '#a78bfa' : '#f43f5e';
                                        const fontSize = Math.max(10, 13 / scale);
                                        const padX = 4 / scale;
                                        const padY = 2 / scale;
                                        const labelH = fontSize + padY * 2;
                                        return (
                                            <React.Fragment key={ann.id}>
                                                <Rect
                                                    x={bx} y={by} width={bw} height={bh}
                                                    stroke={color}
                                                    strokeWidth={1.5 / scale}
                                                    fill={unclassified
                                                        ? 'rgba(245,158,11,0.10)'
                                                        : ann.source === 'auto'
                                                            ? 'rgba(167,139,250,0.07)'
                                                            : 'rgba(244,63,94,0.07)'}
                                                    dash={unclassified ? [6 / scale, 3 / scale] : undefined}
                                                />
                                                {/* Label background */}
                                                <Rect
                                                    x={bx} y={by - labelH}
                                                    width={Math.min(bw, ((unclassified ? 12 : ann.class_name.length) * fontSize * 0.6) + padX * 2 + 30/scale)}
                                                    height={labelH}
                                                    fill={color}
                                                    cornerRadius={2 / scale}
                                                />
                                                <Text
                                                    x={bx + padX}
                                                    y={by - labelH + padY}
                                                    text={unclassified ? '? unclassified' : `${ann.class_name}${ann.source === 'auto' ? ' ✦' : ''}`}
                                                    fontSize={fontSize}
                                                    fontFamily="sans-serif"
                                                    fill="#fff"
                                                    fontStyle="bold"
                                                />

                                                {/* ── Canvas Controls (Only for AI boxes) ── */}
                                                {ann.source !== 'manual' && (
                                                    <Group x={bx + bw - (44 / scale)} y={by + (4 / scale)}>
                                                        {/* Reject Button */}
                                                        <Group 
                                                            onClick={() => handleRejectAnnotation(ann.id)} 
                                                            onTap={() => handleRejectAnnotation(ann.id)}
                                                            onMouseEnter={e => {
                                                                const stage = e.target.getStage();
                                                                stage.container().style.cursor = 'pointer';
                                                            }}
                                                            onMouseLeave={e => {
                                                                const stage = e.target.getStage();
                                                                stage.container().style.cursor = 'crosshair';
                                                            }}
                                                        >
                                                            <Rect
                                                                width={18 / scale} height={18 / scale}
                                                                fill="#f43f5e" cornerRadius={3 / scale}
                                                                shadowBlur={2 / scale}
                                                            />
                                                            <Text
                                                                text="✕" fill="#fff" fontSize={12 / scale}
                                                                x={5 / scale} y={3 / scale} fontStyle="bold"
                                                            />
                                                        </Group>
                                                        {/* Accept Button */}
                                                        <Group 
                                                            x={22 / scale} 
                                                            onClick={() => handleAcceptAnnotation(ann.id)} 
                                                            onTap={() => handleAcceptAnnotation(ann.id)}
                                                            onMouseEnter={e => {
                                                                const stage = e.target.getStage();
                                                                stage.container().style.cursor = 'pointer';
                                                            }}
                                                            onMouseLeave={e => {
                                                                const stage = e.target.getStage();
                                                                stage.container().style.cursor = 'crosshair';
                                                            }}
                                                        >
                                                            <Rect
                                                                width={18 / scale} height={18 / scale}
                                                                fill="#22c55e" cornerRadius={3 / scale}
                                                                shadowBlur={2 / scale}
                                                            />
                                                            <Text
                                                                text="✓" fill="#fff" fontSize={12 / scale}
                                                                x={4 / scale} y={3 / scale} fontStyle="bold"
                                                            />
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
                                            strokeWidth={2 / scale}
                                            dash={[6 / scale, 3 / scale]}
                                            fill="rgba(220,20,60,0.08)"
                                        />
                                    )}
                                </Layer>
                            </Stage>

                            {/* Class picker floats over the canvas */}
                            {pendingAnnotation && (
                                <ClassPicker
                                    classes={localClasses}
                                    usedClasses={allUsedClasses}
                                    onConfirm={handleClassConfirm}
                                    onCancel={handleClassCancel}
                                    remaining={classifyingAnnId ? aiQueueRef.current.length + 1 : 0}
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
                                        <span className="annotation-class">{ann.class_name}</span>
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
                    onClose={() => setShowTrainingPanel(false)}
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
                    onClose={() => { setShowReviewPanel(false); setReviewFilterIds(null); }}
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
                        else setShowMainTrainingPanel(true);
                    }}
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
        </div>
    );
};

export default AnnotationWorkspace;
