import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Stage, Layer, Rect, Line, Text, Group } from 'react-konva';
import useImage from 'use-image';
import { Route, X, Trash2, Check, AlertTriangle, Plus, Square, Minus, ChevronLeft } from 'lucide-react';
import './SequencePanel.css';

import { API_URL } from '../config';

const ORIGIN = API_URL.replace('/api/v1', '');
const STAGE_W = 640;
const STAGE_H = 420;

const STEP_COLORS = ['#dc143c', '#d97706', '#059669', '#2563eb', '#7c3aed', '#db2777', '#0891b2', '#65a30d'];

function BackgroundImage({ src }) {
    const [image] = useImage(src, 'anonymous');
    if (!image) return null;
    const scale = Math.min(STAGE_W / image.width, STAGE_H / image.height);
    return (
        <Group>
            <Rect x={0} y={0} width={STAGE_W} height={STAGE_H} fill="#111318" />
            <Group x={(STAGE_W - image.width * scale) / 2} y={(STAGE_H - image.height * scale) / 2} scaleX={scale} scaleY={scale}>
                <Rect width={image.width} height={image.height} fillPatternImage={image} />
            </Group>
        </Group>
    );
}

export default function SequencePanel({ project, onClose }) {
    const [sequences, setSequences]     = useState([]);
    const [loading, setLoading]         = useState(true);
    const [error, setError]             = useState(null);
    const [successMsg, setSuccessMsg]   = useState(null);

    const [images, setImages]           = useState([]);
    const [refImage, setRefImage]       = useState(null);

    const [editing, setEditing]         = useState(false); // builder open?
    const [seqName, setSeqName]         = useState('');
    const [seqMode, setSeqMode]         = useState('strict');
    const [steps, setSteps]             = useState([]); // {order_index, label, region_type, region_coords, required_class}
    const [drawTool, setDrawTool]       = useState('box'); // 'box' | 'line'
    const [drawing, setDrawing]         = useState(null); // in-progress shape
    const [pendingClass, setPendingClass] = useState('');

    const stageRef = useRef(null);

    // ── Load sequences + images ──────────────────────────────────
    const fetchAll = useCallback(async () => {
        try {
            const [seqRes, imgRes] = await Promise.all([
                axios.get(`${API_URL}/sequences/project/${project.id}`),
                axios.get(`${API_URL}/images/project/${project.id}`),
            ]);
            setSequences(seqRes.data);
            setImages(imgRes.data);
            if (imgRes.data.length && !refImage) setRefImage(imgRes.data[0]);
        } catch {
            setError('Failed to load sequences.');
        }
    }, [project.id, refImage]);

    useEffect(() => {
        setLoading(true);
        fetchAll().finally(() => setLoading(false));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const showSuccess = (msg) => {
        setSuccessMsg(msg);
        setTimeout(() => setSuccessMsg(null), 4000);
    };

    // ── Builder: start / reset ─────────────────────────────────────
    const startNewSequence = () => {
        setSeqName('');
        setSeqMode('strict');
        setSteps([]);
        setPendingClass('');
        setEditing(true);
    };

    // ── Drawing on canvas ───────────────────────────────────────────
    const handleMouseDown = (e) => {
        const pos = e.target.getStage().getPointerPosition();
        setDrawing({ x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y });
    };

    const handleMouseMove = (e) => {
        if (!drawing) return;
        const pos = e.target.getStage().getPointerPosition();
        setDrawing(d => ({ ...d, x2: pos.x, y2: pos.y }));
    };

    const handleMouseUp = () => {
        if (!drawing) return;
        const { x1, y1, x2, y2 } = drawing;
        const dist = Math.hypot(x2 - x1, y2 - y1);
        if (dist < 8) { setDrawing(null); return; } // ignore accidental clicks

        if (!pendingClass.trim()) {
            setError('Set a "required object class" before drawing a region.');
            setDrawing(null);
            return;
        }

        const nx1 = Math.min(x1, x2) / STAGE_W, nx2 = Math.max(x1, x2) / STAGE_W;
        const ny1 = Math.min(y1, y2) / STAGE_H, ny2 = Math.max(y1, y2) / STAGE_H;
        const coords = drawTool === 'line' ? [x1 / STAGE_W, y1 / STAGE_H, x2 / STAGE_W, y2 / STAGE_H] : [nx1, ny1, nx2, ny2];

        setSteps(prev => [...prev, {
            order_index: prev.length,
            label: `Step ${prev.length + 1}`,
            region_type: drawTool,
            region_coords: coords,
            required_class: pendingClass.trim(),
        }]);
        setDrawing(null);
        setError(null);
    };

    const updateStepLabel = (idx, label) =>
        setSteps(prev => prev.map((s, i) => i === idx ? { ...s, label } : s));

    const removeStep = (idx) =>
        setSteps(prev => prev.filter((_, i) => i !== idx).map((s, i) => ({ ...s, order_index: i })));

    const moveStep = (idx, dir) => {
        setSteps(prev => {
            const next = [...prev];
            const target = idx + dir;
            if (target < 0 || target >= next.length) return prev;
            [next[idx], next[target]] = [next[target], next[idx]];
            return next.map((s, i) => ({ ...s, order_index: i }));
        });
    };

    // ── Save / delete sequence ──────────────────────────────────────
    const handleSave = async () => {
        if (!seqName.trim()) { setError('Give the sequence a name.'); return; }
        if (steps.length === 0) { setError('Draw at least one region first.'); return; }
        setError(null);
        try {
            const res = await axios.post(`${API_URL}/sequences/project/${project.id}`, {
                name: seqName.trim(),
                mode: seqMode,
                steps,
            });
            setSequences(prev => [res.data, ...prev]);
            setEditing(false);
            showSuccess(`Sequence "${res.data.name}" saved with ${steps.length} step${steps.length !== 1 ? 's' : ''}.`);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to save sequence.');
        }
    };

    const handleDelete = async (id) => {
        if (!window.confirm('Delete this sequence?')) return;
        try {
            await axios.delete(`${API_URL}/sequences/${id}`);
            setSequences(prev => prev.filter(s => s.id !== id));
            showSuccess('Sequence deleted.');
        } catch {
            setError('Failed to delete sequence.');
        }
    };

    // ── Render ────────────────────────────────────────────────────
    return (
        <div className="sq-overlay" onClick={onClose}>
            <div className="sq-modal" onClick={e => e.stopPropagation()}>

                {/* Header */}
                <div className="sq-header">
                    <div className="sq-header-left">
                        {editing && (
                            <button className="sq-back" onClick={() => setEditing(false)}><ChevronLeft size={18} /></button>
                        )}
                        <span className="sq-header-icon"><Route size={20} /></span>
                        <div>
                            <h2 className="sq-title">Sequence Detection</h2>
                            <p className="sq-subtitle">
                                {editing
                                    ? 'Draw ordered regions a tracked object must pass through, in order'
                                    : 'Define ordered region checkpoints — e.g. a bulb visiting socket 1 → 2 → 3'}
                            </p>
                        </div>
                    </div>
                    <button className="sq-close" onClick={onClose}><X size={18} /></button>
                </div>

                <div className="sq-body">
                    {error && (
                        <div className="sq-error">
                            <span><AlertTriangle size={14} /> {error}</span>
                            <button onClick={() => setError(null)}><X size={14} /></button>
                        </div>
                    )}
                    {successMsg && <div className="sq-success"><Check size={14} /> {successMsg}</div>}

                    {!editing ? (
                        <>
                            <button className="sq-btn-new" onClick={startNewSequence} disabled={images.length === 0}>
                                <Plus size={16} /> New Sequence
                            </button>
                            {images.length === 0 && (
                                <p className="sq-hint">Upload/extract at least one image or video frame first — it's used as the reference frame for drawing regions.</p>
                            )}

                            <div className="sq-list">
                                {loading ? (
                                    <div className="sq-list-empty">Loading…</div>
                                ) : sequences.length === 0 ? (
                                    <div className="sq-list-empty">No sequences yet.</div>
                                ) : (
                                    sequences.map(seq => (
                                        <div key={seq.id} className="sq-card">
                                            <div className="sq-card-info">
                                                <span className="sq-card-name">{seq.name}</span>
                                                <div className="sq-card-meta">
                                                    <span className={`sq-mode-badge sq-mode-badge--${seq.mode}`}>{seq.mode}</span>
                                                    <span>{seq.steps.length} step{seq.steps.length !== 1 ? 's' : ''}</span>
                                                </div>
                                                <div className="sq-card-steps">
                                                    {seq.steps.sort((a, b) => a.order_index - b.order_index).map((s, i) => (
                                                        <React.Fragment key={i}>
                                                            {i > 0 && <span className="sq-step-arrow">→</span>}
                                                            <span className="sq-step-chip" style={{ borderColor: STEP_COLORS[i % STEP_COLORS.length] }}>
                                                                {s.label} <em>({s.required_class})</em>
                                                            </span>
                                                        </React.Fragment>
                                                    ))}
                                                </div>
                                            </div>
                                            <button className="sq-btn-delete" onClick={() => handleDelete(seq.id)} title="Delete sequence">
                                                <Trash2 size={14} />
                                            </button>
                                        </div>
                                    ))
                                )}
                            </div>
                        </>
                    ) : (
                        <div className="sq-builder">
                            <div className="sq-builder-top">
                                <input
                                    className="sq-input"
                                    placeholder="Sequence name (e.g. Bulb Assembly Line)"
                                    value={seqName}
                                    onChange={e => setSeqName(e.target.value)}
                                />
                                <select className="sq-select" value={seqMode} onChange={e => setSeqMode(e.target.value)}>
                                    <option value="strict">Strict — wrong region resets progress</option>
                                    <option value="lenient">Lenient — wrong region is ignored</option>
                                </select>
                                {images.length > 1 && (
                                    <select
                                        className="sq-select"
                                        value={refImage?.id || ''}
                                        onChange={e => setRefImage(images.find(i => i.id === e.target.value))}
                                    >
                                        {images.map(img => <option key={img.id} value={img.id}>{img.filename}</option>)}
                                    </select>
                                )}
                            </div>

                            <div className="sq-builder-main">
                                <div className="sq-canvas-wrap">
                                    <div className="sq-toolbar">
                                        <button
                                            className={`sq-tool-btn ${drawTool === 'box' ? 'sq-tool-btn--active' : ''}`}
                                            onClick={() => setDrawTool('box')}
                                        ><Square size={13} /> Box</button>
                                        <button
                                            className={`sq-tool-btn ${drawTool === 'line' ? 'sq-tool-btn--active' : ''}`}
                                            onClick={() => setDrawTool('line')}
                                        ><Minus size={13} /> Line</button>
                                        <input
                                            className="sq-class-input"
                                            placeholder="required object class (e.g. bulb)"
                                            value={pendingClass}
                                            onChange={e => setPendingClass(e.target.value)}
                                        />
                                    </div>
                                    <Stage
                                        ref={stageRef}
                                        width={STAGE_W}
                                        height={STAGE_H}
                                        className="sq-stage"
                                        onMouseDown={handleMouseDown}
                                        onMouseMove={handleMouseMove}
                                        onMouseUp={handleMouseUp}
                                    >
                                        <Layer>
                                            {refImage && (
                                                <BackgroundImage src={`${ORIGIN}${refImage.filepath}`} />
                                            )}
                                            {steps.map((s, i) => {
                                                const color = STEP_COLORS[i % STEP_COLORS.length];
                                                if (s.region_type === 'box') {
                                                    const [x1, y1, x2, y2] = s.region_coords;
                                                    return (
                                                        <Group key={i}>
                                                            <Rect
                                                                x={x1 * STAGE_W} y={y1 * STAGE_H}
                                                                width={(x2 - x1) * STAGE_W} height={(y2 - y1) * STAGE_H}
                                                                stroke={color} strokeWidth={2} fill={`${color}22`}
                                                            />
                                                            <Text x={x1 * STAGE_W + 4} y={y1 * STAGE_H + 4} text={`${i + 1}. ${s.label}`} fill={color} fontStyle="bold" fontSize={12} />
                                                        </Group>
                                                    );
                                                }
                                                const [x1, y1, x2, y2] = s.region_coords;
                                                return (
                                                    <Group key={i}>
                                                        <Line points={[x1 * STAGE_W, y1 * STAGE_H, x2 * STAGE_W, y2 * STAGE_H]} stroke={color} strokeWidth={3} />
                                                        <Text x={x1 * STAGE_W + 4} y={y1 * STAGE_H - 16} text={`${i + 1}. ${s.label}`} fill={color} fontStyle="bold" fontSize={12} />
                                                    </Group>
                                                );
                                            })}
                                            {drawing && (
                                                drawTool === 'box' ? (
                                                    <Rect
                                                        x={Math.min(drawing.x1, drawing.x2)} y={Math.min(drawing.y1, drawing.y2)}
                                                        width={Math.abs(drawing.x2 - drawing.x1)} height={Math.abs(drawing.y2 - drawing.y1)}
                                                        stroke="#ffffff" strokeWidth={1.5} dash={[4, 4]}
                                                    />
                                                ) : (
                                                    <Line points={[drawing.x1, drawing.y1, drawing.x2, drawing.y2]} stroke="#ffffff" strokeWidth={2} dash={[4, 4]} />
                                                )
                                            )}
                                        </Layer>
                                    </Stage>
                                </div>

                                <div className="sq-steps-panel">
                                    <h4 className="sq-steps-title">Steps (in order)</h4>
                                    {steps.length === 0 ? (
                                        <p className="sq-hint">Set a required class above, then draw a box or line on the frame to add step 1, then step 2, etc.</p>
                                    ) : (
                                        <ol className="sq-steps-list">
                                            {steps.map((s, i) => (
                                                <li key={i} className="sq-step-row">
                                                    <span className="sq-step-dot" style={{ background: STEP_COLORS[i % STEP_COLORS.length] }} />
                                                    <input
                                                        className="sq-step-label-input"
                                                        value={s.label}
                                                        onChange={e => updateStepLabel(i, e.target.value)}
                                                    />
                                                    <span className="sq-step-class">{s.required_class}</span>
                                                    <div className="sq-step-actions">
                                                        <button onClick={() => moveStep(i, -1)} disabled={i === 0} title="Move up">↑</button>
                                                        <button onClick={() => moveStep(i, 1)} disabled={i === steps.length - 1} title="Move down">↓</button>
                                                        <button onClick={() => removeStep(i)} title="Remove"><Trash2 size={12} /></button>
                                                    </div>
                                                </li>
                                            ))}
                                        </ol>
                                    )}
                                </div>
                            </div>

                            <div className="sq-builder-footer">
                                <button className="sq-btn-cancel" onClick={() => setEditing(false)}>Cancel</button>
                                <button className="sq-btn-save" onClick={handleSave}><Check size={15} /> Save Sequence</button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
