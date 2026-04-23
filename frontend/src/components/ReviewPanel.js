import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Stage, Layer, Rect, Text, Image as KonvaImg, Group } from 'react-konva';
import useImage from 'use-image';
import './ReviewPanel.css';
import { Check, X, Sparkles, ChevronLeft, ChevronRight } from 'lucide-react';
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

export default function ReviewPanel({ project, images, onClose, onAnnotationsUpdated, filterImageIds }) {
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
    const stageWrapRef = useRef(null);

    const currentImage = reviewImages[currentIdx] || null;

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

    // Reset loaded size when navigating to a different image
    useEffect(() => {
        setLoadedImageSize(null);
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

    // Canvas drawing
    const handleMouseDown = (e) => {
        if (pendingAnnotation) return;
        const pos = e.target.getStage().getPointerPosition();
        setIsDrawing(true);
        setNewAnnotation({ x: pos.x / scale, y: pos.y / scale, width: 0, height: 0 });
    };

    const handleMouseMove = (e) => {
        if (!isDrawing) return;
        const pos = e.target.getStage().getPointerPosition();
        setNewAnnotation(prev => ({
            ...prev,
            width: pos.x / scale - prev.x,
            height: pos.y / scale - prev.y,
        }));
    };

    const handleMouseUp = () => {
        setIsDrawing(false);
        if (!newAnnotation || Math.abs(newAnnotation.width) < 5 || Math.abs(newAnnotation.height) < 5) {
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

    // Use browser-reported natural dimensions (EXIF-corrected) once loaded
    const imgW = loadedImageSize?.width  || currentImage?.width  || 1;
    const imgH = loadedImageSize?.height || currentImage?.height || 1;

    const scale = currentImage
        ? Math.min(1, canvasSize.w / imgW, canvasSize.h / imgH)
        : 1;
    const stageW = currentImage ? Math.round(imgW * scale) : canvasSize.w;
    const stageH = currentImage ? Math.round(imgH * scale) : canvasSize.h;

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
                    <p>Run Auto-Annotate first, then come back to review the results here.</p>
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
                            <div className="rp-header-title">Review Annotations</div>
                            <div className="rp-header-sub">{project.name}</div>
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

                    <button
                        className="rp-exit-btn"
                        onClick={() => { onAnnotationsUpdated?.(); onClose(); }}
                    >
                        <X size={16} /> Exit Review
                    </button>
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
                                        <span className="rp-canvas-dims">
                                            {imgW} × {imgH}px
                                        </span>
                                        <span className="rp-canvas-hint">
                                            Draw a box to add new annotation
                                        </span>
                                        <span className="rp-ann-count-badge">
                                            {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
                                            {autoCount > 0 && (
                                                <span className="rp-unverified-tag">{autoCount} unverified</span>
                                            )}
                                        </span>
                                    </div>

                                    <div className="rp-stage-wrap" ref={stageWrapRef}>
                                        <Stage
                                            width={stageW}
                                            height={stageH}
                                            scaleX={scale}
                                            scaleY={scale}
                                            style={{ cursor: 'crosshair' }}
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
                                                    const color = isAuto ? '#a78bfa' : '#22c55e';
                                                    const fill  = isAuto
                                                        ? 'rgba(167,139,250,0.10)'
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

                                                            {/* Inline accept/reject on canvas for auto boxes */}
                                                            {isAuto && (
                                                                <Group
                                                                    x={bx + bw - 44 / scale}
                                                                    y={by + 4 / scale}
                                                                >
                                                                    <Group
                                                                        onClick={() => handleRejectAnnotation(ann.id)}
                                                                        onTap={() => handleRejectAnnotation(ann.id)}
                                                                        onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer'; }}
                                                                        onMouseLeave={e => { e.target.getStage().container().style.cursor = 'crosshair'; }}
                                                                    >
                                                                        <Rect width={18 / scale} height={18 / scale} fill="#f43f5e" cornerRadius={3 / scale} />
                                                                        <Text text="✕" fill="#fff" fontSize={12 / scale} x={5 / scale} y={3 / scale} fontStyle="bold" />
                                                                    </Group>
                                                                    <Group
                                                                        x={22 / scale}
                                                                        onClick={() => handleAcceptAnnotation(ann.id)}
                                                                        onTap={() => handleAcceptAnnotation(ann.id)}
                                                                        onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer'; }}
                                                                        onMouseLeave={e => { e.target.getStage().container().style.cursor = 'crosshair'; }}
                                                                    >
                                                                        <Rect width={18 / scale} height={18 / scale} fill="#22c55e" cornerRadius={3 / scale} />
                                                                        <Text text="✓" fill="#fff" fontSize={12 / scale} x={4 / scale} y={3 / scale} fontStyle="bold" />
                                                                    </Group>
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
                                                        strokeWidth={2 / scale}
                                                        dash={[6 / scale, 3 / scale]}
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
                                    const dotColor = isAuto ? '#a78bfa' : '#22c55e';
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
                                                <span className="rp-verified-tag">verified</span>
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
