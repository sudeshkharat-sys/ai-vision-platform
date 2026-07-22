import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Stage, Layer, Rect, Text, Image as KonvaImg, Group } from 'react-konva';
import useImage from 'use-image';
import './ReviewPanel.css';
import { Check, X, Sparkles, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Maximize2, Trash2 } from 'lucide-react';
import logoImg from '../logo.png';

import { API_URL, BASE_URL } from '../config';

const CanvasImage = ({ src, onLoad }) => {
    const [img] = useImage(src, 'anonymous');
    useEffect(() => {
        if (img && onLoad) {
            onLoad(img.naturalWidth || img.width, img.naturalHeight || img.height);
        }
    }, [img, onLoad]);
    return <KonvaImg image={img} />;
};

const ClassPicker = ({ classes, onConfirm, onCancel }) => {
    const [custom, setCustom] = useState('');
    const inputRef = useRef(null);

    useEffect(() => {
        if (classes.length === 0 && inputRef.current) inputRef.current.focus();
    }, []); // eslint-disable-line

    return (
        <div className="rp-class-overlay" onClick={onCancel}>
            <div className="rp-class-picker" onClick={e => e.stopPropagation()}>
                <div className="rp-class-header">
                    <span>Select Class</span>
                    <button onClick={onCancel}><X size={14} /></button>
                </div>
                {classes.length > 0 && (
                    <div className="rp-class-list">
                        {classes.map(cls => (
                            <button key={cls} className="rp-class-item" onClick={() => onConfirm(cls)}>
                                <span className="rp-class-dot" />
                                {cls}
                            </button>
                        ))}
                    </div>
                )}
                <div className="rp-class-custom">
                    <input
                        ref={inputRef}
                        value={custom}
                        onChange={e => setCustom(e.target.value)}
                        placeholder="Custom class name…"
                        onKeyDown={e => { if (e.key === 'Enter' && custom.trim()) onConfirm(custom.trim()); }}
                    />
                    <button onClick={() => custom.trim() && onConfirm(custom.trim())}>Add</button>
                </div>
            </div>
        </div>
    );
};

export default function ReviewPanel({ project, images, onClose, onAnnotationsUpdated, filterImageIds, filterLabel, filterClassName }) {
    // If filterImageIds is provided (e.g. from AL suggestions), show only those images.
    // Otherwise show the normal annotated/annotating review queue.
    const reviewImages = filterImageIds?.size > 0
        ? images.filter(img => filterImageIds.has(String(img.id)))
        : images.filter(img => img.status === 'annotated' || img.status === 'annotating');

    const [currentIdx, setCurrentIdx] = useState(0);
    const [annotations, setAnnotations] = useState([]);
    const [loading, setLoading] = useState(false);
    // Track which image IDs the user has explicitly acted on
    const [reviewedIds, setReviewedIds] = useState(new Set());
    const [canvasSize, setCanvasSize] = useState({ w: 700, h: 480 });
    const [isDrawing, setIsDrawing] = useState(false);
    const [newAnnotation, setNewAnnotation] = useState(null);
    const [pendingAnnotation, setPendingAnnotation] = useState(null);
    const [statusMsg, setStatusMsg] = useState(null);
    // Actual displayed dimensions after EXIF orientation is applied by the browser
    const [loadedImageSize, setLoadedImageSize] = useState(null);
    // Zoom / pan
    const [userZoom, setUserZoom] = useState(1);
    const [stagePos, setStagePos] = useState({ x: 0, y: 0 });
    // Cache of { auto, total } annotation counts per image id (built as user browses)
    const [annCountCache, setAnnCountCache] = useState({});
    const stageWrapRef = useRef(null);

    const currentImage = reviewImages[currentIdx] || null;

    // Declared early so zoom/draw handlers can reference them without TDZ
    const imgW = loadedImageSize?.width  || currentImage?.width  || 1;
    const imgH = loadedImageSize?.height || currentImage?.height || 1;
    const scale = currentImage ? Math.min(1, canvasSize.w / imgW, canvasSize.h / imgH) : 1;
    const stageW = currentImage ? Math.round(imgW * scale) : canvasSize.w;
    const stageH = currentImage ? Math.round(imgH * scale) : canvasSize.h;

    // Flash a short status message
    const showStatus = useCallback((msg) => {
        setStatusMsg(msg);
        setTimeout(() => setStatusMsg(null), 1800);
    }, []);

    // Measure the rp-stage-wrap container directly to avoid image cropping
    // rp-stage-wrap has padding: 12px on all sides, so subtract 24px each axis
    const measureCanvas = useCallback(() => {
        if (stageWrapRef.current) {
            const r = stageWrapRef.current.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
                setCanvasSize({ w: Math.max(200, r.width - 24), h: Math.max(150, r.height - 24) });
            }
        }
    }, []);

    useEffect(() => {
        measureCanvas();
        window.addEventListener('resize', measureCanvas);
        return () => window.removeEventListener('resize', measureCanvas);
    }, [measureCanvas]);

    // Re-measure when image changes so stageWrapRef is populated
    useEffect(() => {
        measureCanvas();
    }, [currentImage?.id, measureCanvas]);

    const handleImageLoad = useCallback((w, h) => {
        setLoadedImageSize({ width: w, height: h });
    }, []);

    // Reset loaded size, zoom, and pan when navigating to a different image
    useEffect(() => {
        setLoadedImageSize(null);
        setUserZoom(1);
        setStagePos({ x: 0, y: 0 });
        setPendingAnnotation(null);
        setNewAnnotation(null);
        setIsDrawing(false);
    }, [currentImage?.id]);

    // Load annotations when current image changes
    useEffect(() => {
        if (!currentImage) return;
        setLoading(true);
        setAnnotations([]);
        axios.get(`${API_URL}/annotations/image/${currentImage.id}`)
            .then(res => setAnnotations(res.data))
            .catch(() => setAnnotations([]))
            .finally(() => setLoading(false));
    }, [currentImage?.id]); // eslint-disable-line

    // Update per-image annotation count cache whenever annotations change
    useEffect(() => {
        if (!currentImage) return;
        const auto = annotations.filter(a => a.source !== 'manual').length;
        setAnnCountCache(prev => ({ ...prev, [currentImage.id]: { auto, total: annotations.length } }));
    }, [annotations, currentImage?.id]); // eslint-disable-line

    const navigate = useCallback((dir) => {
        setCurrentIdx(prev => {
            const next = prev + dir;
            if (next >= 0 && next < reviewImages.length) {
                setPendingAnnotation(null);
                setNewAnnotation(null);
                return next;
            }
            return prev;
        });
    }, [reviewImages.length]);

    const markReviewed = useCallback(() => {
        if (currentImage) {
            setReviewedIds(prev => new Set([...prev, currentImage.id]));
        }
    }, [currentImage]);

    const handleAcceptAnnotation = async (annId) => {
        try {
            await axios.patch(`${API_URL}/annotations/${annId}/verify`);
            setAnnotations(prev => prev.map(a => a.id === annId ? { ...a, source: 'manual' } : a));
            markReviewed();
        } catch {
            // silent
        }
    };

    const handleRejectAnnotation = async (annId) => {
        try {
            await axios.delete(`${API_URL}/annotations/${annId}`);
            setAnnotations(prev => prev.filter(a => a.id !== annId));
            markReviewed();
        } catch {
            // silent
        }
    };

    const handleAcceptAll = useCallback(async () => {
        const toVerify = annotations.filter(a => a.source !== 'manual');
        if (!toVerify.length) return;
        try {
            await Promise.all(toVerify.map(a => axios.patch(`${API_URL}/annotations/${a.id}/verify`)));
            setAnnotations(prev => prev.map(a => ({ ...a, source: 'manual' })));
            markReviewed();
            showStatus(`✓ Accepted ${toVerify.length} annotation${toVerify.length !== 1 ? 's' : ''}`);
        } catch {
            // silent
        }
    }, [annotations, markReviewed, showStatus]);

    const handleRejectAll = useCallback(async () => {
        const toDelete = annotations.filter(a => a.source !== 'manual');
        if (!toDelete.length) return;
        try {
            await Promise.all(toDelete.map(a => axios.delete(`${API_URL}/annotations/${a.id}`)));
            setAnnotations(prev => prev.filter(a => a.source === 'manual'));
            markReviewed();
            showStatus(`✕ Removed ${toDelete.length} auto annotation${toDelete.length !== 1 ? 's' : ''}`);
        } catch {
            // silent
        }
    }, [annotations, markReviewed, showStatus]);

    const handleDeleteAllAnnotations = useCallback(async () => {
        if (!annotations.length) return;
        if (!window.confirm(`Delete all ${annotations.length} annotation(s) on this image? This cannot be undone.`)) return;
        try {
            await Promise.all(annotations.map(a => axios.delete(`${API_URL}/annotations/${a.id}`)));
            setAnnotations([]);
            markReviewed();
            showStatus('All annotations deleted');
        } catch {
            // silent
        }
    }, [annotations, markReviewed, showStatus]);

    const handleDeleteAllProjectAnnotations = useCallback(async () => {
        if (!window.confirm(
            `Delete ALL annotations across every image in "${project.name}"?\n\nThis will reset all ${reviewImages.length} images back to unannotated. This cannot be undone.`
        )) return;
        try {
            const res = await axios.delete(`${API_URL}/projects/${project.id}/annotations`);
            setAnnotations([]);
            onAnnotationsUpdated?.();
            showStatus(`✕ Deleted ${res.data.deleted} annotations across all images`);
        } catch {
            showStatus('Failed to delete annotations');
        }
    }, [project, reviewImages.length, onAnnotationsUpdated, showStatus]);

    // Bulk-remove just the boxes of the filtered class across every image in
    // this edit-mode view. Other classes' boxes on the same images are kept —
    // clears the way to redraw fresh boxes for this class (or a renamed one).
    const handleDeleteClassAnnotations = useCallback(async () => {
        if (!filterClassName) return;
        if (!window.confirm(
            `Delete every "${filterClassName}" box across all ${reviewImages.length} image(s) shown here?\n\nOther boxes on these images are kept. This cannot be undone.`
        )) return;
        try {
            const res = await axios.delete(
                `${API_URL}/projects/${project.id}/class-annotations/${encodeURIComponent(filterClassName)}`
            );
            setAnnotations(prev => prev.filter(a => a.class_name !== filterClassName));
            setAnnCountCache({});
            onAnnotationsUpdated?.();
            showStatus(`✕ Deleted ${res.data.deleted_count} "${filterClassName}" box(es)`);
        } catch {
            showStatus('Failed to delete annotations');
        }
    }, [filterClassName, project, reviewImages.length, onAnnotationsUpdated, showStatus]);

    // Keyboard shortcuts
    useEffect(() => {
        const handler = (e) => {
            if (pendingAnnotation) return;
            const tag = document.activeElement?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA') return;

            if (e.key === 'ArrowRight') { e.preventDefault(); navigate(1); }
            if (e.key === 'ArrowLeft')  { e.preventDefault(); navigate(-1); }
            if (e.key === 'e' || e.key === 'E') handleAcceptAll();
            if (e.key === 'r' || e.key === 'R') handleRejectAll();
            if (e.key === 'Escape') { onAnnotationsUpdated?.(); onClose(); }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [pendingAnnotation, navigate, handleAcceptAll, handleRejectAll, onClose, onAnnotationsUpdated]);

    // Zoom via mouse wheel — zoom around the cursor point
    const handleWheel = (e) => {
        e.evt.preventDefault();
        const stage = e.target.getStage();
        const pointer = stage.getPointerPosition();
        const scaleBy = 1.12;
        const oldZoom = userZoom;
        const newZoom = Math.max(0.2, Math.min(15, e.evt.deltaY < 0 ? oldZoom * scaleBy : oldZoom / scaleBy));
        const cx = Math.round((canvasSize.w - stageW) / 2);
        const cy = Math.round((canvasSize.h - stageH) / 2);
        const totalX = cx + stagePos.x;
        const totalY = cy + stagePos.y;
        const mousePointTo = {
            x: (pointer.x - totalX) / (scale * oldZoom),
            y: (pointer.y - totalY) / (scale * oldZoom),
        };
        setUserZoom(newZoom);
        setStagePos({
            x: pointer.x - mousePointTo.x * scale * newZoom - cx,
            y: pointer.y - mousePointTo.y * scale * newZoom - cy,
        });
    };

    const resetZoom = () => { setUserZoom(1); setStagePos({ x: 0, y: 0 }); };

    const handleMouseDown = (e) => {
        if (pendingAnnotation) return;
        const cls = e.target.getClassName ? e.target.getClassName() : '';
        if (cls === 'Rect' || cls === 'Group' || cls === 'Text') return;
        const pos = e.target.getStage().getRelativePointerPosition();
        setIsDrawing(true);
        setNewAnnotation({ x: pos.x, y: pos.y, width: 0, height: 0 });
    };

    const handleMouseMove = (e) => {
        if (!isDrawing) return;
        const pos = e.target.getStage().getRelativePointerPosition();
        setNewAnnotation(prev => ({
            ...prev,
            width: pos.x - prev.x,
            height: pos.y - prev.y,
        }));
    };

    const handleMouseUp = () => {
        setIsDrawing(false);
        if (!newAnnotation || Math.abs(newAnnotation.width) < 5 / userZoom || Math.abs(newAnnotation.height) < 5 / userZoom) {
            setNewAnnotation(null);
            return;
        }
        const x = newAnnotation.width < 0 ? newAnnotation.x + newAnnotation.width : newAnnotation.x;
        const y = newAnnotation.height < 0 ? newAnnotation.y + newAnnotation.height : newAnnotation.y;
        setPendingAnnotation({ x, y, width: Math.abs(newAnnotation.width), height: Math.abs(newAnnotation.height) });
        setNewAnnotation(null);
    };

    const handleClassConfirm = async (className) => {
        if (!pendingAnnotation || !currentImage) return;
        const { x, y, width: w, height: h } = pendingAnnotation;
        const bbox = [
            (x + w / 2) / imgW,
            (y + h / 2) / imgH,
            w / imgW,
            h / imgH,
        ];
        try {
            const res = await axios.post(`${API_URL}/annotations`, {
                image_id: currentImage.id,
                class_name: className,
                bbox,
                source: 'manual',
            });
            setAnnotations(prev => [...prev, res.data]);
            showStatus(`+ Added "${className}"`);
        } catch {
            // silent
        }
        setPendingAnnotation(null);
    };

    const autoCount = annotations.filter(a => a.source !== 'manual').length;
    const drawnBox = pendingAnnotation || newAnnotation;
    const progressPct = reviewImages.length > 0 ? (reviewedIds.size / reviewImages.length) * 100 : 0;

    // ── Empty state ─────────────────────────────────────────────
    if (reviewImages.length === 0) {
        return (
            <div className="rp-overlay">
                <div className="rp-empty-card">
                    <div className="rp-empty-icon"><Check size={32} /></div>
                    <h3>Nothing to Review</h3>
                    <p>
                        {filterLabel
                            ? 'None of these images could be loaded — they may have changed since the class list was refreshed.'
                            : 'Run Auto-Annotate first, then come back to review the results here.'}
                    </p>
                    <button className="rp-btn rp-btn-neutral" onClick={onClose}>Close</button>
                </div>
            </div>
        );
    }

    return (
        <div className="rp-overlay">
            <div className="rp-shell">

                {/* ── Header ── */}
                <div className="rp-header">
                    <div className="rp-header-left">
                        <span className="rp-header-icon"><Sparkles size={20} /></span>
                        <div>
                            <div className="rp-header-title">
                                {filterLabel ? 'Edit Mode' : 'Review Annotations'}
                            </div>
                            <div className="rp-header-sub">
                                {project.name}{filterLabel ? ` — ${filterLabel}` : ''}
                            </div>
                        </div>
                    </div>

                    <div className="rp-header-center">
                        <div className="rp-progress-track">
                            <div className="rp-progress-fill" style={{ width: `${progressPct}%` }} />
                        </div>
                        <span className="rp-progress-label">
                            {reviewedIds.size} / {reviewImages.length} reviewed
                        </span>
                    </div>

                    <div className="rp-header-right">
                        {filterClassName && (
                            <button
                                className="rp-delete-all-btn rp-delete-class-btn"
                                onClick={handleDeleteClassAnnotations}
                                title={`Delete every "${filterClassName}" box across the images shown here — other classes are kept`}
                            >
                                <Trash2 size={15} /> Delete All "{filterClassName}" Boxes
                            </button>
                        )}
                        <button
                            className="rp-delete-all-btn"
                            onClick={handleDeleteAllProjectAnnotations}
                            title="Delete every annotation across all images in this project"
                        >
                            <Trash2 size={15} /> Delete All Annotations
                        </button>
                        <button
                            className="rp-exit-btn"
                            onClick={() => { onAnnotationsUpdated?.(); onClose(); }}
                        >
                            <X size={16} /> Exit Review
                        </button>
                    </div>
                </div>

                {/* ── Body ── */}
                <div className="rp-body">

                    {/* Image strip (left) */}
                    <div className="rp-strip">
                        <div className="rp-strip-label">Images ({reviewImages.length})</div>
                        {reviewImages.map((img, idx) => (
                            <div
                                key={img.id}
                                className={[
                                    'rp-strip-item',
                                    idx === currentIdx ? 'active' : '',
                                    reviewedIds.has(img.id) ? 'done' : '',
                                ].join(' ')}
                                onClick={() => {
                                    setCurrentIdx(idx);
                                    setPendingAnnotation(null);
                                    setNewAnnotation(null);
                                }}
                                title={img.filename}
                            >
                                <div className="rp-strip-thumb">
                                    <img
                                        src={`${BASE_URL}${img.filepath}`}
                                        alt={img.filename}
                                        crossOrigin="anonymous"
                                    />
                                    {reviewedIds.has(img.id) && (
                                        <div className="rp-strip-check"><Check size={12} /></div>
                                    )}
                                    {annCountCache[img.id]?.auto > 0 && (
                                        <div className="rp-strip-unverified">{annCountCache[img.id].auto}</div>
                                    )}
                                    {annCountCache[img.id]?.auto === 0 && annCountCache[img.id]?.total > 0 && (
                                        <div className="rp-strip-allverified">✓</div>
                                    )}
                                </div>
                                <div className="rp-strip-name">{img.filename}</div>
                            </div>
                        ))}
                    </div>

                    {/* Canvas + annotation panel */}
                    <div className="rp-main">

                        {/* Canvas */}
                        <div className="rp-canvas-area">
                            {statusMsg && <div className="rp-toast">{statusMsg}</div>}

                            {loading && <div className="rp-loading-msg">Loading…</div>}

                            {currentImage && !loading && (
                                <>
                                    <div className="rp-canvas-toolbar">
                                        <span className="rp-canvas-filename">{currentImage.filename}</span>
                                        <span className="rp-canvas-dims">{imgW} × {imgH}px</span>
                                        <span className="rp-canvas-hint">
                                            {'Draw a box to add annotation · scroll to zoom'}
                                        </span>
                                        <button className="rp-zoom-btn" onClick={() => setUserZoom(z => Math.min(15, z * 1.3))} title="Zoom in"><ZoomIn size={14} /></button>
                                        <span className="rp-zoom-pct">{Math.round(userZoom * 100)}%</span>
                                        <button className="rp-zoom-btn" onClick={() => setUserZoom(z => Math.max(0.2, z / 1.3))} title="Zoom out"><ZoomOut size={14} /></button>
                                        <button className="rp-zoom-btn" onClick={resetZoom} title="Reset zoom"><Maximize2 size={14} /></button>
                                        <span className="rp-ann-count-badge">
                                            {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
                                            {autoCount > 0 && (
                                                <span className="rp-unverified-tag">{autoCount} unverified</span>
                                            )}
                                        </span>
                                    </div>

                                    <div className="rp-stage-wrap" ref={stageWrapRef} style={{ overflow: 'hidden', cursor: 'crosshair' }}>
                                        <Stage
                                            width={canvasSize.w}
                                            height={canvasSize.h}
                                            scaleX={scale * userZoom}
                                            scaleY={scale * userZoom}
                                            x={Math.round((canvasSize.w - stageW) / 2) + stagePos.x}
                                            y={Math.round((canvasSize.h - stageH) / 2) + stagePos.y}
                                            draggable={false}
                                            onWheel={handleWheel}
                                            onDragEnd={(e) => {
                                                const cx = Math.round((canvasSize.w - stageW) / 2);
                                                const cy = Math.round((canvasSize.h - stageH) / 2);
                                                setStagePos({ x: e.target.x() - cx, y: e.target.y() - cy });
                                            }}
                                            onMouseDown={handleMouseDown}
                                            onMouseMove={handleMouseMove}
                                            onMouseUp={handleMouseUp}
                                        >
                                            <Layer>
                                                <CanvasImage
                                                    src={`${BASE_URL}${currentImage.filepath}`}
                                                    onLoad={handleImageLoad}
                                                />

                                                {annotations.map(ann => {
                                                    const bx = (ann.bbox[0] - ann.bbox[2] / 2) * imgW;
                                                    const by = (ann.bbox[1] - ann.bbox[3] / 2) * imgH;
                                                    const bw = ann.bbox[2] * imgW;
                                                    const bh = ann.bbox[3] * imgH;
                                                    const isAuto = ann.source !== 'manual';
                                                    const color = isAuto ? '#f59e0b' : '#22c55e';
                                                    const fill  = isAuto
                                                        ? 'rgba(245,158,11,0.10)'
                                                        : 'rgba(34,197,94,0.07)';
                                                    const fs   = Math.max(10, 13 / scale);
                                                    const padX = 4 / scale;
                                                    const padY = 2 / scale;
                                                    const lh   = fs + padY * 2;
                                                    const labelW = Math.min(
                                                        bw,
                                                        ann.class_name.length * fs * 0.6 + padX * 2
                                                    );

                                                    return (
                                                        <React.Fragment key={ann.id}>
                                                            <Rect
                                                                x={bx} y={by}
                                                                width={bw} height={bh}
                                                                stroke={color}
                                                                strokeWidth={isAuto ? 2 / scale : 1.5 / scale}
                                                                dash={isAuto ? [6 / scale, 3 / scale] : undefined}
                                                                fill={fill}
                                                            />
                                                            {/* Label pill */}
                                                            <Rect
                                                                x={bx} y={by - lh}
                                                                width={labelW} height={lh}
                                                                fill={color}
                                                                cornerRadius={2 / scale}
                                                            />
                                                            <Text
                                                                x={bx + padX} y={by - lh + padY}
                                                                text={`${ann.class_name}${isAuto ? ' ✦' : ''}`}
                                                                fontSize={fs}
                                                                fontFamily="sans-serif"
                                                                fill="#fff"
                                                                fontStyle="bold"
                                                            />

                                                            {/* Inline buttons on canvas */}
                                                            {isAuto ? (
                                                                <Group
                                                                    x={bx + bw - 44 / scale}
                                                                    y={by + 4 / scale}
                                                                >
                                                                    <Group
                                                                        onClick={() => handleRejectAnnotation(ann.id)}
                                                                        onTap={() => handleRejectAnnotation(ann.id)}
                                                                        onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer'; }}
                                                                        onMouseLeave={e => { e.target.getStage().container().style.cursor = userZoom > 1 ? 'grab' : 'crosshair'; }}
                                                                    >
                                                                        <Rect width={18 / scale} height={18 / scale} fill="#f43f5e" cornerRadius={3 / scale} />
                                                                        <Text text="✕" fill="#fff" fontSize={12 / scale} x={5 / scale} y={3 / scale} fontStyle="bold" />
                                                                    </Group>
                                                                    <Group
                                                                        x={22 / scale}
                                                                        onClick={() => handleAcceptAnnotation(ann.id)}
                                                                        onTap={() => handleAcceptAnnotation(ann.id)}
                                                                        onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer'; }}
                                                                        onMouseLeave={e => { e.target.getStage().container().style.cursor = userZoom > 1 ? 'grab' : 'crosshair'; }}
                                                                    >
                                                                        <Rect width={18 / scale} height={18 / scale} fill="#22c55e" cornerRadius={3 / scale} />
                                                                        <Text text="✓" fill="#fff" fontSize={12 / scale} x={4 / scale} y={3 / scale} fontStyle="bold" />
                                                                    </Group>
                                                                </Group>
                                                            ) : (
                                                                /* Delete button on canvas for verified annotations */
                                                                <Group
                                                                    x={bx + bw - 22 / scale}
                                                                    y={by + 4 / scale}
                                                                    onClick={() => handleRejectAnnotation(ann.id)}
                                                                    onTap={() => handleRejectAnnotation(ann.id)}
                                                                    onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer'; }}
                                                                    onMouseLeave={e => { e.target.getStage().container().style.cursor = userZoom > 1 ? 'grab' : 'crosshair'; }}
                                                                >
                                                                    <Rect width={18 / scale} height={18 / scale} fill="#f43f5e" cornerRadius={3 / scale} />
                                                                    <Text text="✕" fill="#fff" fontSize={12 / scale} x={5 / scale} y={3 / scale} fontStyle="bold" />
                                                                </Group>
                                                            )}
                                                        </React.Fragment>
                                                    );
                                                })}

                                                {drawnBox && (
                                                    <Rect
                                                        x={drawnBox.x} y={drawnBox.y}
                                                        width={drawnBox.width} height={drawnBox.height}
                                                        stroke="#dc143c"
                                                        strokeWidth={2 / (scale * userZoom)}
                                                        dash={[6 / (scale * userZoom), 3 / (scale * userZoom)]}
                                                        fill="rgba(220,20,60,0.08)"
                                                    />
                                                )}
                                            </Layer>
                                        </Stage>

                                        {pendingAnnotation && (
                                            <ClassPicker
                                                classes={project.classes || []}
                                                onConfirm={handleClassConfirm}
                                                onCancel={() => setPendingAnnotation(null)}
                                            />
                                        )}
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Annotation review panel (right) */}
                        <div className="rp-ann-panel">
                            <div className="rp-ann-panel-header">
                                <span className="rp-ann-panel-title">Annotations</span>
                                {autoCount > 0 && (
                                    <span className="rp-unverified-pill">{autoCount} to review</span>
                                )}
                            </div>

                            <div className="rp-ann-list">
                                {!loading && annotations.length === 0 && (
                                    <div className="rp-ann-empty">
                                        No annotations on this image.<br />
                                        <span className="rp-ann-empty-hint">Draw a box to add one.</span>
                                    </div>
                                )}

                                {annotations.map(ann => {
                                    const isAuto = ann.source !== 'manual';
                                    const dotColor = isAuto ? '#f59e0b' : '#22c55e';
                                    return (
                                        <div
                                            key={ann.id}
                                            className={`rp-ann-row ${isAuto ? 'auto' : 'verified'}`}
                                        >
                                            <span className="rp-ann-dot" style={{ background: dotColor }} />
                                            <span className="rp-ann-class">{ann.class_name}</span>
                                            <span className={`rp-source-tag src-${ann.source}`}>
                                                {ann.source === 'ai_prompt' ? 'AI' : ann.source}
                                            </span>
                                            {isAuto ? (
                                                <div className="rp-ann-row-actions">
                                                    <button
                                                        className="rp-row-btn keep"
                                                        title="Keep this annotation"
                                                        onClick={() => handleAcceptAnnotation(ann.id)}
                                                    >
                                                        <Check size={14} /> Keep
                                                    </button>
                                                    <button
                                                        className="rp-row-btn remove"
                                                        title="Remove this annotation"
                                                        onClick={() => handleRejectAnnotation(ann.id)}
                                                    >
                                                        <X size={14} /> Remove
                                                    </button>
                                                </div>
                                            ) : (
                                                <div className="rp-ann-row-actions">
                                                    <span className="rp-verified-tag">verified</span>
                                                    <button
                                                        className="rp-row-btn remove"
                                                        title="Delete this verified annotation"
                                                        onClick={() => handleRejectAnnotation(ann.id)}
                                                    >
                                                        <X size={14} />
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Bulk actions */}
                            <div className="rp-bulk">
                                <button
                                    className="rp-bulk-btn rp-bulk-accept"
                                    disabled={autoCount === 0}
                                    onClick={handleAcceptAll}
                                    title="Accept all auto annotations (E)"
                                >
                                    <Check size={14} /> Accept All Auto ({autoCount})
                                </button>
                                <button
                                    className="rp-bulk-btn rp-bulk-reject"
                                    disabled={autoCount === 0}
                                    onClick={handleRejectAll}
                                    title="Reject all auto annotations (R)"
                                >
                                    <X size={14} /> Reject All Auto
                                </button>
                                <button
                                    className="rp-bulk-btn rp-bulk-delete-all"
                                    disabled={annotations.length === 0}
                                    onClick={handleDeleteAllAnnotations}
                                    title="Delete every annotation on this image (including verified)"
                                >
                                    <Trash2 size={14} /> Delete All ({annotations.length})
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ── Footer ── */}
                <div className="rp-footer">
                    <div className="rp-footer-shortcuts">
                        <span className="rp-kbd">← →</span> Navigate
                        <span className="rp-kbd">E</span> Accept All
                        <span className="rp-kbd">R</span> Reject All
                        <span className="rp-kbd">Esc</span> Exit
                    </div>

                    <div className="rp-footer-counter">
                        {currentIdx + 1} / {reviewImages.length}
                    </div>

                    <div className="rp-footer-nav">
                        <button
                            className="rp-nav-btn"
                            disabled={currentIdx === 0}
                            onClick={() => navigate(-1)}
                        >
                            <ChevronLeft size={16} /> Prev
                        </button>
                        <button
                            className="rp-nav-btn rp-nav-next"
                            disabled={currentIdx === reviewImages.length - 1}
                            onClick={() => {
                                markReviewed();
                                navigate(1);
                            }}
                        >
                            Next <ChevronRight size={16} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
