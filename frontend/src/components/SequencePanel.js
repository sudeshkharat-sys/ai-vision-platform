import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Stage, Layer, Rect, Line, Text, Group, Image as KonvaImage } from 'react-konva';
import useImage from 'use-image';
import { Route, X, Trash2, Check, AlertTriangle, Plus, Square, Minus, ChevronLeft, MousePointerClick, Play, Loader2 } from 'lucide-react';
import './SequencePanel.css';

import { API_URL } from '../config';

const ORIGIN = API_URL.replace('/api/v1', '');
const STAGE_W = 640;
const STAGE_H = 420;

const STEP_COLORS = ['#dc143c', '#d97706', '#059669', '#2563eb', '#7c3aed', '#db2777', '#0891b2', '#65a30d'];
let _regionSeq = 0;
const nextRegionId = () => `r${Date.now()}_${_regionSeq++}`;

function BackgroundImage({ src }) {
    const [image] = useImage(src, 'anonymous');
    if (!image) return null;
    const scale = Math.min(STAGE_W / image.width, STAGE_H / image.height);
    return (
        <Group listening={false}>
            <Rect x={0} y={0} width={STAGE_W} height={STAGE_H} fill="#111318" />
            <Group x={(STAGE_W - image.width * scale) / 2} y={(STAGE_H - image.height * scale) / 2} scaleX={scale} scaleY={scale}>
                <KonvaImage image={image} width={image.width} height={image.height} />
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
    const [videos, setVideos]           = useState([]);

    // Per-sequence run state: { [seqId]: { videoId, run } }
    const [runByCard, setRunByCard]     = useState({});
    const pollRef = useRef({});

    const [editing, setEditing]         = useState(false); // builder open?
    const [seqName, setSeqName]         = useState('');
    const [seqMode, setSeqMode]         = useState('strict');

    // Regions are the unique shapes drawn on the frame (a keyboard has one region per key).
    // stepOrder is the ordered sequence of region ids to visit — a region id CAN repeat
    // (e.g. the "S" key region is visited twice when spelling "SUDESH").
    const [regions, setRegions]         = useState([]); // {id, region_type, region_coords, required_class, label}
    const [stepOrder, setStepOrder]     = useState([]); // [regionId, regionId, ...]

    const [drawTool, setDrawTool]       = useState('box'); // 'box' | 'line'
    const [drawing, setDrawing]         = useState(null); // in-progress shape
    const [pendingClass, setPendingClass] = useState('');
    const [pendingLabel, setPendingLabel] = useState('');

    const stageRef = useRef(null);

    // ── Load sequences + images ──────────────────────────────────
    const fetchAll = useCallback(async () => {
        try {
            const [seqRes, imgRes, vidRes] = await Promise.all([
                axios.get(`${API_URL}/sequences/project/${project.id}`),
                axios.get(`${API_URL}/images/project/${project.id}`),
                axios.get(`${API_URL}/videos/project/${project.id}`),
            ]);
            setSequences(seqRes.data);
            setImages(imgRes.data);
            setVideos(vidRes.data);
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
        setRegions([]);
        setStepOrder([]);
        setPendingClass('');
        setPendingLabel('');
        setEditing(true);
    };

    // ── Drawing a NEW region on canvas ───────────────────────────────
    const handleMouseDown = (e) => {
        // Only start a fresh draw when clicking empty canvas, not an existing region
        if (e.target !== e.target.getStage()) return;
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
            setError('Set a "required object class" before drawing a new region.');
            setDrawing(null);
            return;
        }

        const nx1 = Math.min(x1, x2) / STAGE_W, nx2 = Math.max(x1, x2) / STAGE_W;
        const ny1 = Math.min(y1, y2) / STAGE_H, ny2 = Math.max(y1, y2) / STAGE_H;
        const coords = drawTool === 'line' ? [x1 / STAGE_W, y1 / STAGE_H, x2 / STAGE_W, y2 / STAGE_H] : [nx1, ny1, nx2, ny2];

        const id = nextRegionId();
        const label = pendingLabel.trim() || `Region ${regions.length + 1}`;
        setRegions(prev => [...prev, {
            id,
            region_type: drawTool,
            region_coords: coords,
            required_class: pendingClass.trim(),
            label,
        }]);
        setStepOrder(prev => [...prev, id]);
        setDrawing(null);
        setError(null);
        setPendingLabel('');
    };

    // ── Click an EXISTING region to add it again as the next step ────
    const appendExistingRegion = (regionId) => {
        setStepOrder(prev => [...prev, regionId]);
    };

    const removeStepAt = (idx) =>
        setStepOrder(prev => prev.filter((_, i) => i !== idx));

    const moveStep = (idx, dir) => {
        setStepOrder(prev => {
            const next = [...prev];
            const target = idx + dir;
            if (target < 0 || target >= next.length) return prev;
            [next[idx], next[target]] = [next[target], next[idx]];
            return next;
        });
    };

    const updateRegionLabel = (regionId, label) =>
        setRegions(prev => prev.map(r => r.id === regionId ? { ...r, label } : r));

    const deleteRegion = (regionId) => {
        setRegions(prev => prev.filter(r => r.id !== regionId));
        setStepOrder(prev => prev.filter(id => id !== regionId));
    };

    // ── Save / delete sequence ──────────────────────────────────────
    const handleSave = async () => {
        if (!seqName.trim()) { setError('Give the sequence a name.'); return; }
        if (stepOrder.length === 0) { setError('Draw at least one region and add it to the sequence.'); return; }
        setError(null);
        try {
            const steps = stepOrder.map((regionId, i) => {
                const r = regions.find(reg => reg.id === regionId);
                return {
                    order_index: i,
                    label: r.label,
                    region_type: r.region_type,
                    region_coords: r.region_coords,
                    required_class: r.required_class,
                };
            });
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

    const regionColor = (regionId) => {
        const idx = regions.findIndex(r => r.id === regionId);
        return STEP_COLORS[idx % STEP_COLORS.length];
    };

    // ── Run a sequence against a video ──────────────────────────────
    useEffect(() => () => Object.values(pollRef.current).forEach(clearInterval), []);

    const setCardVideo = (seqId, videoId) =>
        setRunByCard(prev => ({ ...prev, [seqId]: { ...(prev[seqId] || {}), videoId } }));

    const pollRun = (seqId, runId) => {
        if (pollRef.current[seqId]) clearInterval(pollRef.current[seqId]);
        pollRef.current[seqId] = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/sequences/runs/${runId}`);
                setRunByCard(prev => ({ ...prev, [seqId]: { ...(prev[seqId] || {}), run: res.data } }));
                if (['complete', 'error', 'failed'].includes(res.data.status)) {
                    clearInterval(pollRef.current[seqId]);
                    delete pollRef.current[seqId];
                }
            } catch {
                clearInterval(pollRef.current[seqId]);
                delete pollRef.current[seqId];
            }
        }, 2000);
    };

    const startRun = async (seq) => {
        const videoId = runByCard[seq.id]?.videoId || videos[0]?.id;
        if (!videoId) { setError('Upload a video for this project first.'); return; }
        setError(null);
        try {
            const res = await axios.post(`${API_URL}/sequences/${seq.id}/run/${videoId}`);
            setRunByCard(prev => ({ ...prev, [seq.id]: { videoId, run: res.data } }));
            pollRun(seq.id, res.data.id);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to start the run.');
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
                                    ? 'Draw regions, then click one to add it to the sequence — a region can be reused (e.g. a key pressed twice)'
                                    : 'Define ordered region checkpoints — e.g. hand-gate 1 → 2 → 3, then key S → U → D → E → S → H'}
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
                                                    {[...seq.steps].sort((a, b) => a.order_index - b.order_index).map((s, i) => (
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

                            {sequences.length > 0 && (
                                <div className="sq-run-section">
                                    <h4 className="sq-steps-title">Run against a video</h4>
                                    {videos.length === 0 ? (
                                        <p className="sq-hint">Import a video for this project first (sidebar → Import Video), then come back here to run a sequence against it.</p>
                                    ) : (
                                        sequences.map(seq => {
                                            const card = runByCard[seq.id] || {};
                                            const run = card.run;
                                            const isRunning = run && !['complete', 'error', 'failed'].includes(run.status);
                                            return (
                                                <div key={seq.id} className="sq-run-card">
                                                    <div className="sq-run-row">
                                                        <span className="sq-run-name">{seq.name}</span>
                                                        <select
                                                            className="sq-select sq-run-video-select"
                                                            value={card.videoId || videos[0]?.id || ''}
                                                            onChange={e => setCardVideo(seq.id, e.target.value)}
                                                            disabled={isRunning}
                                                        >
                                                            {videos.map(v => <option key={v.id} value={v.id}>{v.original_filename}</option>)}
                                                        </select>
                                                        <button className="sq-btn-run" onClick={() => startRun(seq)} disabled={isRunning}>
                                                            {isRunning ? <Loader2 size={13} className="sq-spin" /> : <Play size={13} />}
                                                            {isRunning ? 'Running…' : 'Run'}
                                                        </button>
                                                    </div>
                                                    {run && (
                                                        <div className="sq-run-result">
                                                            <div className="sq-run-progress">
                                                                <span className={`sq-run-status sq-run-status--${run.status}`}>
                                                                    {run.status === 'complete' && run.passed ? '✓ Passed' :
                                                                     run.status === 'complete' ? '✗ Incomplete' :
                                                                     run.status === 'error' ? '⚠ Error' :
                                                                     run.status}
                                                                </span>
                                                                <span className="sq-run-step-count">step {run.current_step} / {run.total_steps}</span>
                                                            </div>
                                                            {run.error && <p className="sq-run-error">{run.error}</p>}
                                                            {run.step_events.length > 0 && (
                                                                <div className="sq-run-events">
                                                                    {run.step_events.map((ev, i) => (
                                                                        <span key={i} className={`sq-run-event sq-run-event--${ev.reason}`}>
                                                                            {ev.label} · frame {ev.frame_number}
                                                                            {ev.reason === 'matched' ? ' ✓' : ev.reason === 'wrong_region_reset' ? ' ✗ reset' : ' (ignored)'}
                                                                        </span>
                                                                    ))}
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })
                                    )}
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="sq-builder">
                            <div className="sq-builder-top">
                                <input
                                    className="sq-input"
                                    placeholder="Sequence name (e.g. Type SUDESH on virtual keyboard)"
                                    value={seqName}
                                    onChange={e => setSeqName(e.target.value)}
                                />
                                <select className="sq-select" value={seqMode} onChange={e => setSeqMode(e.target.value)}>
                                    <option value="strict">Strict — wrong region resets progress</option>
                                    <option value="lenient">Lenient — wrong region is ignored</option>
                                </select>
                            </div>

                            {images.length > 1 && (
                                <div className="sq-ref-picker">
                                    <span className="sq-ref-picker-label">Reference frame</span>
                                    <div className="sq-ref-thumbs">
                                        {images.map(img => (
                                            <button
                                                type="button"
                                                key={img.id}
                                                className={`sq-ref-thumb ${refImage?.id === img.id ? 'sq-ref-thumb--active' : ''}`}
                                                onClick={() => setRefImage(img)}
                                                title={`${img.filename} — ${img.width}×${img.height}`}
                                            >
                                                <img
                                                    src={`${ORIGIN}${img.filepath}?w=${img.width}&h=${img.height}`}
                                                    alt={img.filename}
                                                    loading="lazy"
                                                />
                                                <span className="sq-ref-thumb-dims">{img.width}×{img.height}</span>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

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
                                            placeholder="region name (e.g. S key, Gate 1)"
                                            value={pendingLabel}
                                            onChange={e => setPendingLabel(e.target.value)}
                                        />
                                        <input
                                            className="sq-class-input"
                                            placeholder="required object class (e.g. finger)"
                                            value={pendingClass}
                                            onChange={e => setPendingClass(e.target.value)}
                                        />
                                    </div>
                                    <p className="sq-hint sq-hint--tight">
                                        <MousePointerClick size={12} /> Draw a new region on empty canvas. Click an already-drawn region to reuse it as the next step (e.g. the "S" key a second time).
                                    </p>
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
                                                <BackgroundImage src={`${ORIGIN}${refImage.filepath}?w=${refImage.width}&h=${refImage.height}`} />
                                            )}
                                            {regions.map((r, i) => {
                                                const color = STEP_COLORS[i % STEP_COLORS.length];
                                                const timesUsed = stepOrder.filter(id => id === r.id).length;
                                                if (r.region_type === 'box') {
                                                    const [x1, y1, x2, y2] = r.region_coords;
                                                    return (
                                                        <Group
                                                            key={r.id}
                                                            onClick={() => appendExistingRegion(r.id)}
                                                            onTap={() => appendExistingRegion(r.id)}
                                                        >
                                                            <Rect
                                                                x={x1 * STAGE_W} y={y1 * STAGE_H}
                                                                width={(x2 - x1) * STAGE_W} height={(y2 - y1) * STAGE_H}
                                                                stroke={color} strokeWidth={2} fill={`${color}22`}
                                                            />
                                                            <Text x={x1 * STAGE_W + 4} y={y1 * STAGE_H + 4} text={`${r.label}${timesUsed > 1 ? ` ×${timesUsed}` : ''}`} fill={color} fontStyle="bold" fontSize={12} />
                                                        </Group>
                                                    );
                                                }
                                                const [x1, y1, x2, y2] = r.region_coords;
                                                return (
                                                    <Group
                                                        key={r.id}
                                                        onClick={() => appendExistingRegion(r.id)}
                                                        onTap={() => appendExistingRegion(r.id)}
                                                    >
                                                        <Line points={[x1 * STAGE_W, y1 * STAGE_H, x2 * STAGE_W, y2 * STAGE_H]} stroke={color} strokeWidth={3} hitStrokeWidth={16} />
                                                        <Text x={x1 * STAGE_W + 4} y={y1 * STAGE_H - 16} text={`${r.label}${timesUsed > 1 ? ` ×${timesUsed}` : ''}`} fill={color} fontStyle="bold" fontSize={12} />
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
                                    <h4 className="sq-steps-title">Sequence steps (in order)</h4>
                                    {stepOrder.length === 0 ? (
                                        <p className="sq-hint">Set a name + required class above, then draw a region to add step 1. Draw more regions, or click an existing one to repeat it.</p>
                                    ) : (
                                        <ol className="sq-steps-list">
                                            {stepOrder.map((regionId, i) => {
                                                const r = regions.find(reg => reg.id === regionId);
                                                if (!r) return null;
                                                return (
                                                    <li key={`${regionId}-${i}`} className="sq-step-row">
                                                        <span className="sq-step-dot" style={{ background: regionColor(regionId) }} />
                                                        <span className="sq-step-num">{i + 1}.</span>
                                                        <span className="sq-step-region-label">{r.label}</span>
                                                        <span className="sq-step-class">{r.required_class}</span>
                                                        <div className="sq-step-actions">
                                                            <button onClick={() => moveStep(i, -1)} disabled={i === 0} title="Move up">↑</button>
                                                            <button onClick={() => moveStep(i, 1)} disabled={i === stepOrder.length - 1} title="Move down">↓</button>
                                                            <button onClick={() => removeStepAt(i)} title="Remove from sequence"><Trash2 size={12} /></button>
                                                        </div>
                                                    </li>
                                                );
                                            })}
                                        </ol>
                                    )}

                                    {regions.length > 0 && (
                                        <>
                                            <h4 className="sq-steps-title sq-steps-title--spaced">Drawn regions</h4>
                                            <div className="sq-region-chip-list">
                                                {regions.map((r, i) => (
                                                    <div key={r.id} className="sq-region-chip" style={{ borderColor: STEP_COLORS[i % STEP_COLORS.length] }}>
                                                        <input
                                                            className="sq-region-chip-input"
                                                            value={r.label}
                                                            onChange={e => updateRegionLabel(r.id, e.target.value)}
                                                        />
                                                        <button className="sq-region-chip-add" onClick={() => appendExistingRegion(r.id)} title="Add to sequence">+</button>
                                                        <button className="sq-region-chip-del" onClick={() => deleteRegion(r.id)} title="Delete region"><Trash2 size={11} /></button>
                                                    </div>
                                                ))}
                                            </div>
                                        </>
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
