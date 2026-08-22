import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import axios from 'axios';
import { Stage, Layer, Rect, Line, Text, Group, Image as KonvaImage } from 'react-konva';
import useImage from 'use-image';
import { Route, X, Trash2, Check, AlertTriangle, Plus, Square, Minus, ChevronLeft, MousePointerClick, Play, Loader2, Download } from 'lucide-react';
import './SequencePanel.css';

import { API_URL } from '../config';

const ORIGIN = API_URL.replace('/api/v1', '');
const STAGE_W = 640;
const STAGE_H = 420;

const STEP_COLORS = ['#dc143c', '#d97706', '#059669', '#2563eb', '#7c3aed', '#db2777', '#0891b2', '#65a30d'];
let _regionSeq = 0;
const nextRegionId = () => `r${Date.now()}_${_regionSeq++}`;

// Where the reference photo actually sits inside the fixed 640x420 canvas —
// scaled to fit and centered, so a landscape or portrait photo doesn't fill
// the whole canvas. Everything that turns a canvas click into a stored
// region (and back) must use THIS rect, not the raw canvas size, or a
// region drawn near the edge of a non-640:420 photo ends up recorded in
// the wrong place relative to the actual image content.
function computeImageLayout(imgWidth, imgHeight) {
    if (!imgWidth || !imgHeight) {
        return { x: 0, y: 0, width: STAGE_W, height: STAGE_H };
    }
    const scale = Math.min(STAGE_W / imgWidth, STAGE_H / imgHeight);
    const width = imgWidth * scale;
    const height = imgHeight * scale;
    return { x: (STAGE_W - width) / 2, y: (STAGE_H - height) / 2, width, height };
}

function BackgroundImage({ src, layout }) {
    const [image] = useImage(src, 'anonymous');
    if (!image) return null;
    return (
        <Group listening={false}>
            <Rect x={0} y={0} width={STAGE_W} height={STAGE_H} fill="#111318" />
            <Group x={layout.x} y={layout.y} scaleX={layout.width / image.width} scaleY={layout.height / image.height}>
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
    const [availableClasses, setAvailableClasses] = useState([]); // real classes annotated in this project (count > 0)

    // Per-sequence run state: { [seqId]: { videoId, run } }
    const [runByCard, setRunByCard]     = useState({});
    const pollRef = useRef({});

    const [editing, setEditing]         = useState(false); // builder open?
    const [seqName, setSeqName]         = useState('');
    const [seqMode, setSeqMode]         = useState('strict');

    // Regions are the unique shapes drawn on the frame (a keyboard has one region per key).
    // stepOrder is the ordered sequence of region ids to visit — a region id CAN repeat
    // (e.g. the "A" checkpoint region is visited twice in a repeated pattern).
    const [regions, setRegions]         = useState([]); // {id, region_type, region_coords, required_class, label}
    const [stepOrder, setStepOrder]     = useState([]); // [regionId, regionId, ...]

    // What kind of target the NEXT step being added is:
    // "box" / "line" — a region you draw on the canvas.
    // "class" — target = another detected class's own mask (e.g. "M"),
    //   no drawing needed, just pick classes.
    const [regionKind, setRegionKind]   = useState('box');
    const [drawing, setDrawing]         = useState(null); // in-progress shape
    const [pendingClass, setPendingClass] = useState('');
    const [pendingLabel, setPendingLabel] = useState('');
    const [pendingTargetClass, setPendingTargetClass] = useState('');
    const [pendingCompleteOn, setPendingCompleteOn] = useState('detect'); // 'detect' | 'detect_hold' | 'undetect_hold'
    const [pendingHoldSeconds, setPendingHoldSeconds] = useState(0.4);
    const [pendingFreezeBoundary, setPendingFreezeBoundary] = useState(true);
    const [freezing, setFreezing] = useState(false);
    const [seqThreshold, setSeqThreshold] = useState(0.5);

    // Quick single-image test — no save, no video, instant per-step check
    const [testing, setTesting]         = useState(false);
    const [testResults, setTestResults] = useState(null); // { results: [...] } | { error }

    const stageRef = useRef(null);

    // The reference photo's actual displayed rect inside the fixed-size
    // canvas — used to convert every mouse click into a position relative
    // to the PHOTO, not the canvas frame around it.
    const imgLayout = useMemo(
        () => computeImageLayout(refImage?.width, refImage?.height),
        [refImage]
    );

    // Stale test results are worse than none — clear them whenever the
    // steps or reference image change so nobody trusts an outdated pass/fail.
    useEffect(() => { setTestResults(null); }, [stepOrder, refImage]);

    // ── Load sequences + images + the project's actual annotated classes ──
    const fetchAll = useCallback(async () => {
        try {
            const [seqRes, imgRes, vidRes, statsRes] = await Promise.all([
                axios.get(`${API_URL}/sequences/project/${project.id}`),
                axios.get(`${API_URL}/images/project/${project.id}`),
                axios.get(`${API_URL}/videos/project/${project.id}`),
                axios.get(`${API_URL}/pipeline/training-stats/${project.id}`).catch(() => ({ data: {} })),
            ]);
            setSequences(seqRes.data);
            setImages(imgRes.data);
            setVideos(vidRes.data);
            const breakdown = statsRes.data?.class_breakdown || {};
            setAvailableClasses(Object.keys(breakdown).filter(c => breakdown[c] > 0).sort());
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
        if (regionKind === 'class') return; // "Class" steps aren't drawn
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

        const classes = pendingClass.split(',').map(c => c.trim()).filter(Boolean);
        if (classes.length === 0) {
            setError('Set a "required object class" before drawing a new region — comma-separate two or more to require all of them at once (e.g. "hand, m").');
            setDrawing(null);
            return;
        }

        // Convert canvas pixel coords -> position relative to the PHOTO
        // itself (imgLayout), not the raw 640x420 canvas frame around it —
        // clamped to 0-1 since a drag can end in the letterboxed padding.
        const toImgX = px => Math.min(1, Math.max(0, (px - imgLayout.x) / imgLayout.width));
        const toImgY = py => Math.min(1, Math.max(0, (py - imgLayout.y) / imgLayout.height));

        const nx1 = toImgX(Math.min(x1, x2)), nx2 = toImgX(Math.max(x1, x2));
        const ny1 = toImgY(Math.min(y1, y2)), ny2 = toImgY(Math.max(y1, y2));
        const coords = regionKind === 'line' ? [toImgX(x1), toImgY(y1), toImgX(x2), toImgY(y2)] : [nx1, ny1, nx2, ny2];

        const id = nextRegionId();
        const label = pendingLabel.trim() || `Region ${regions.length + 1}`;
        setRegions(prev => [...prev, {
            id,
            target_type: 'region',
            region_type: regionKind,
            region_coords: coords,
            required_class: classes[0],
            required_classes: classes,
            label,
        }]);
        setStepOrder(prev => [...prev, id]);
        setDrawing(null);
        setError(null);
        setPendingLabel('');
    };

    // ── Pick a class from the project's real annotated classes instead
    // of typing it — toggles it in/out of the comma-separated field ────
    const toggleIntersectionClass = (cls) => {
        const current = pendingClass.split(',').map(c => c.trim()).filter(Boolean);
        const next = current.includes(cls) ? current.filter(c => c !== cls) : [...current, cls];
        setPendingClass(next.join(', '));
    };

    // ── Add a "Detection Class" step — target = another class's own
    // detected mask (e.g. "M"), no region drawn on canvas. By default the
    // boundary is frozen once (from the reference image) into a fixed
    // polygon region, so the step keeps working even when something (a
    // finger) fully covers/occludes that class later and it can't be
    // live-detected anymore. Uncheck "freeze" to keep the old live-match
    // behavior (target_type: detection_class, re-detected every frame). ─
    const addDetectionClassStep = async () => {
        const targetClass = pendingTargetClass.trim();
        const classes = pendingClass.split(',').map(c => c.trim()).filter(Boolean);
        if (!targetClass) { setError('Select a target class first.'); return; }
        if (classes.length === 0) { setError('Select at least one intersection class first.'); return; }

        const id = nextRegionId();
        const label = pendingLabel.trim() || `${targetClass} x ${classes.join('+')}`;
        const base = {
            id,
            required_class: classes[0],
            required_classes: classes,
            label,
            complete_on: pendingCompleteOn,
            hold_seconds: pendingCompleteOn !== 'detect' ? Number(pendingHoldSeconds) || 0.4 : undefined,
        };

        if (pendingFreezeBoundary) {
            if (!refImage) { setError('Pick a reference frame first — freezing needs it to read the class boundary from.'); return; }
            setFreezing(true);
            setError(null);
            try {
                const res = await axios.get(`${API_URL}/sequences/freeze-class/${project.id}`, {
                    params: { image_id: refImage.id, class_name: targetClass },
                });
                setRegions(prev => [...prev, {
                    ...base,
                    target_type: 'region',
                    region_type: 'polygon',
                    region_coords: res.data.polygon,
                }]);
                setStepOrder(prev => [...prev, id]);
                setPendingLabel('');
            } catch (err) {
                setError(err.response?.data?.detail || `Could not freeze "${targetClass}"'s boundary — make sure it's visible (unoccluded) on the reference frame.`);
            } finally {
                setFreezing(false);
            }
            return;
        }

        setRegions(prev => [...prev, {
            ...base,
            target_type: 'detection_class',
            target_class: targetClass,
        }]);
        setStepOrder(prev => [...prev, id]);
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

    // ── Save / test / delete sequence ─────────────────────────────────
    const buildStepsPayload = () => stepOrder.map((regionId, i) => {
        const r = regions.find(reg => reg.id === regionId);
        return {
            order_index: i,
            label: r.label,
            target_type: r.target_type || 'region',
            region_type: r.region_type,
            region_coords: r.region_coords,
            target_class: r.target_class,
            required_class: r.required_class,
            required_classes: r.required_classes && r.required_classes.length > 1 ? r.required_classes : undefined,
            complete_on: r.complete_on === 'undetect_hold' ? 'undetect_hold' : undefined,
            hold_seconds: r.complete_on === 'undetect_hold' ? r.hold_seconds : undefined,
        };
    });

    const handleSave = async () => {
        if (!seqName.trim()) { setError('Give the sequence a name.'); return; }
        if (stepOrder.length === 0) { setError('Draw at least one region and add it to the sequence.'); return; }
        setError(null);
        try {
            const steps = buildStepsPayload();
            const res = await axios.post(`${API_URL}/sequences/project/${project.id}`, {
                name: seqName.trim(),
                mode: seqMode,
                overlap_threshold: seqThreshold,
                steps,
                ref_image_width: refImage?.width,
                ref_image_height: refImage?.height,
            });
            setSequences(prev => [res.data, ...prev]);
            setEditing(false);
            showSuccess(`Sequence "${res.data.name}" saved with ${steps.length} step${steps.length !== 1 ? 's' : ''}.`);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to save sequence.');
        }
    };

    const handleExportSequence = (seq) => {
        const blob = new Blob([JSON.stringify(seq, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${seq.name.replace(/[^a-z0-9_-]+/gi, '_')}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };

    const handleTestOnImage = async () => {
        if (!refImage) { setError('Pick a reference frame to test against first.'); return; }
        if (stepOrder.length === 0) { setError('Draw at least one region first.'); return; }
        setError(null);
        setTesting(true);
        setTestResults(null);
        try {
            const steps = buildStepsPayload();
            const res = await axios.post(`${API_URL}/sequences/test-image/${project.id}`, {
                image_id: refImage.id,
                steps,
                overlap_threshold: seqThreshold,
            });
            setTestResults(res.data);
        } catch (err) {
            setTestResults({ error: err.response?.data?.detail || 'Test failed.' });
        } finally {
            setTesting(false);
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
                                    : 'Define ordered checkpoints — e.g. Checkpoint A → B → C, then class X → Y → Z'}
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
                                                                {s.label} <em>({(s.required_classes && s.required_classes.length > 1) ? s.required_classes.join(' + ') : s.required_class})</em>
                                                                {s.complete_on === 'undetect_hold' && <em> · gone {s.hold_seconds || 1}s</em>}
                                                                {s.complete_on === 'detect_hold' && <em> · hold {s.hold_seconds || 1}s</em>}
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
                                                        <button
                                                            className="sq-btn-export"
                                                            onClick={() => handleExportSequence(seq)}
                                                            title="Download this sequence as JSON — use it with scripts/sequence_viewer.py to watch it run live on your own machine"
                                                        >
                                                            <Download size={13} /> Export
                                                        </button>
                                                    </div>
                                                    {run && (
                                                        <div className="sq-run-result">
                                                            {run.status === 'running' && run.latest_frame_url && (
                                                                <div className="sq-run-live">
                                                                    <img
                                                                        className="sq-run-live-img"
                                                                        src={`${ORIGIN}${run.latest_frame_url}?f=${run.latest_frame_number}`}
                                                                        alt="Live processing preview"
                                                                    />
                                                                    <span className="sq-run-live-badge">● live · frame {run.latest_frame_number}</span>
                                                                </div>
                                                            )}
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
                                                                        <div key={i} className={`sq-run-event-card sq-run-event-card--${ev.reason}`}>
                                                                            {ev.frame_url && (
                                                                                <a href={`${ORIGIN}${ev.frame_url}`} target="_blank" rel="noopener noreferrer">
                                                                                    <img className="sq-run-event-thumb" src={`${ORIGIN}${ev.frame_url}`} alt={`${ev.label} at frame ${ev.frame_number}`} />
                                                                                </a>
                                                                            )}
                                                                            <span className={`sq-run-event sq-run-event--${ev.reason}`}>
                                                                                {ev.label} · frame {ev.frame_number}
                                                                                {ev.reason === 'matched' ? ' ✓' : ev.reason === 'wrong_region_reset' ? ' ✗ reset' : ' (ignored)'}
                                                                            </span>
                                                                        </div>
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
                                    placeholder="Sequence name"
                                    value={seqName}
                                    onChange={e => setSeqName(e.target.value)}
                                />
                                <select className="sq-select" value={seqMode} onChange={e => setSeqMode(e.target.value)}>
                                    <option value="strict">Strict — wrong region resets progress</option>
                                    <option value="lenient">Lenient — wrong region is ignored</option>
                                </select>
                                <label className="sq-threshold-field" title="How much of the detected object's box must overlap a region to count as 'on it'">
                                    Overlap ≥
                                    <input
                                        type="number" min="0.1" max="1" step="0.05"
                                        className="sq-threshold-input"
                                        value={seqThreshold}
                                        onChange={e => setSeqThreshold(Math.min(1, Math.max(0.1, parseFloat(e.target.value) || 0.5)))}
                                    />
                                </label>
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
                                            className={`sq-tool-btn ${regionKind === 'box' ? 'sq-tool-btn--active' : ''}`}
                                            onClick={() => setRegionKind('box')}
                                            title="Draw a box region on the reference frame"
                                        ><Square size={13} /> Box</button>
                                        <button
                                            className={`sq-tool-btn ${regionKind === 'line' ? 'sq-tool-btn--active' : ''}`}
                                            onClick={() => setRegionKind('line')}
                                            title="Draw a line region on the reference frame"
                                        ><Minus size={13} /> Line</button>
                                        <button
                                            className={`sq-tool-btn ${regionKind === 'class' ? 'sq-tool-btn--active' : ''}`}
                                            onClick={() => setRegionKind('class')}
                                            title="Target = another detected class's own mask — no drawing needed"
                                        >Class</button>

                                        {regionKind === 'class' ? (
                                            <select
                                                className="sq-class-input"
                                                value={pendingTargetClass}
                                                onChange={e => setPendingTargetClass(e.target.value)}
                                            >
                                                <option value="">Select target class…</option>
                                                {availableClasses.map(cls => <option key={cls} value={cls}>{cls}</option>)}
                                            </select>
                                        ) : (
                                            <input
                                                className="sq-class-input"
                                                placeholder="region name"
                                                value={pendingLabel}
                                                onChange={e => setPendingLabel(e.target.value)}
                                            />
                                        )}
                                        <input
                                            className="sq-class-input"
                                            placeholder={regionKind === 'class'
                                                ? 'intersect with class(es) — pick below, or type comma-separated'
                                                : 'required class(es) — pick below, or type comma-separated'}
                                            value={pendingClass}
                                            onChange={e => setPendingClass(e.target.value)}
                                        />
                                        {regionKind === 'class' && (
                                            <>
                                                <label className="sq-class-input" style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                                                    <input
                                                        type="checkbox"
                                                        checked={pendingFreezeBoundary}
                                                        onChange={e => setPendingFreezeBoundary(e.target.checked)}
                                                    />
                                                    Freeze boundary (survives occlusion)
                                                </label>
                                                <select
                                                    className="sq-class-input"
                                                    value={pendingCompleteOn}
                                                    onChange={e => setPendingCompleteOn(e.target.value)}
                                                    title="How this step decides it's done"
                                                >
                                                    <option value="detect">Pass instantly on touch (gate-style)</option>
                                                    <option value="detect_hold">Press & hold — must stay on for N seconds (real key press, ignores pass-through)</option>
                                                    <option value="undetect_hold">Gone for N seconds (gesture released/removed)</option>
                                                </select>
                                                {pendingCompleteOn !== 'detect' && (
                                                    <input
                                                        className="sq-class-input"
                                                        type="number"
                                                        min="0.1"
                                                        step="0.1"
                                                        style={{ width: 70 }}
                                                        title={pendingCompleteOn === 'detect_hold'
                                                            ? 'Seconds the class(es) must stay continuously matched before this step passes'
                                                            : 'Seconds the class(es) must stay undetected before this step passes'}
                                                        value={pendingHoldSeconds}
                                                        onChange={e => setPendingHoldSeconds(e.target.value)}
                                                    />
                                                )}
                                                <button className="sq-btn-add-step" onClick={addDetectionClassStep} disabled={freezing}>
                                                    <Plus size={13} /> {freezing ? 'Freezing…' : 'Add Step'}
                                                </button>
                                            </>
                                        )}
                                    </div>
                                    {availableClasses.length > 0 && (
                                        <div className="sq-class-picker">
                                            {availableClasses.map(cls => {
                                                const active = pendingClass.split(',').map(c => c.trim()).includes(cls);
                                                return (
                                                    <button
                                                        key={cls}
                                                        type="button"
                                                        className={`sq-class-chip ${active ? 'sq-class-chip--active' : ''}`}
                                                        onClick={() => toggleIntersectionClass(cls)}
                                                    >{cls}</button>
                                                );
                                            })}
                                        </div>
                                    )}
                                    <p className="sq-hint sq-hint--tight">
                                        <MousePointerClick size={12} />
                                        {regionKind === 'class'
                                            ? ' Pick a target class + intersect class(es), then click "Add Step" — nothing to draw.'
                                            : ' Draw a new region on empty canvas. Click an already-drawn region to reuse it as the next step. Pick two or more classes to require ALL of them at once.'}
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
                                                <BackgroundImage
                                                    src={`${ORIGIN}${refImage.filepath}?w=${refImage.width}&h=${refImage.height}`}
                                                    layout={imgLayout}
                                                />
                                            )}
                                            {regions.filter(r => r.target_type !== 'detection_class').map((r, i) => {
                                                const color = STEP_COLORS[i % STEP_COLORS.length];
                                                const timesUsed = stepOrder.filter(id => id === r.id).length;
                                                // Region coords are stored relative to the PHOTO (0-1) — convert
                                                // back to canvas pixels via the same imgLayout used to draw it.
                                                const toPx  = fx => imgLayout.x + fx * imgLayout.width;
                                                const toPy  = fy => imgLayout.y + fy * imgLayout.height;
                                                if (r.region_type === 'box') {
                                                    const [x1, y1, x2, y2] = r.region_coords;
                                                    return (
                                                        <Group
                                                            key={r.id}
                                                            onClick={() => appendExistingRegion(r.id)}
                                                            onTap={() => appendExistingRegion(r.id)}
                                                        >
                                                            <Rect
                                                                x={toPx(x1)} y={toPy(y1)}
                                                                width={toPx(x2) - toPx(x1)} height={toPy(y2) - toPy(y1)}
                                                                stroke={color} strokeWidth={2} fill={`${color}22`}
                                                            />
                                                            <Text x={toPx(x1) + 4} y={toPy(y1) + 4} text={`${r.label}${timesUsed > 1 ? ` ×${timesUsed}` : ''}`} fill={color} fontStyle="bold" fontSize={12} />
                                                        </Group>
                                                    );
                                                }
                                                if (r.region_type === 'polygon') {
                                                    // Frozen class boundary — region_coords is a list of [x,y]
                                                    // points (the class's own mask), not a fixed-length tuple.
                                                    const pts = r.region_coords.flatMap(([px, py]) => [toPx(px), toPy(py)]);
                                                    const [fx0, fy0] = r.region_coords[0];
                                                    return (
                                                        <Group
                                                            key={r.id}
                                                            onClick={() => appendExistingRegion(r.id)}
                                                            onTap={() => appendExistingRegion(r.id)}
                                                        >
                                                            <Line points={pts} closed stroke={color} strokeWidth={2} fill={`${color}22`} />
                                                            <Text x={toPx(fx0) + 4} y={toPy(fy0) + 4} text={`${r.label}${timesUsed > 1 ? ` ×${timesUsed}` : ''} 🔒`} fill={color} fontStyle="bold" fontSize={12} />
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
                                                        <Line points={[toPx(x1), toPy(y1), toPx(x2), toPy(y2)]} stroke={color} strokeWidth={3} hitStrokeWidth={16} />
                                                        <Text x={toPx(x1) + 4} y={toPy(y1) - 16} text={`${r.label}${timesUsed > 1 ? ` ×${timesUsed}` : ''}`} fill={color} fontStyle="bold" fontSize={12} />
                                                    </Group>
                                                );
                                            })}
                                            {drawing && (
                                                regionKind === 'box' ? (
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
                                                        <span className="sq-step-class">{(r.required_classes && r.required_classes.length > 1) ? r.required_classes.join(' + ') : r.required_class}</span>
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

                                    {testResults && (
                                        <>
                                            <h4 className="sq-steps-title sq-steps-title--spaced">Test result (single image, no order/motion check)</h4>
                                            {testResults.error ? (
                                                <p className="sq-hint" style={{ color: '#dc2626' }}>{testResults.error}</p>
                                            ) : (
                                                <ol className="sq-steps-list">
                                                    {testResults.results.map((r, i) => (
                                                        <li key={i} className="sq-test-row">
                                                            <span className={`sq-test-badge ${!r.testable ? 'sq-test-badge--na' : r.passed ? 'sq-test-badge--pass' : 'sq-test-badge--fail'}`}>
                                                                {!r.testable ? '—' : r.passed ? '✓' : '✕'}
                                                            </span>
                                                            <span className="sq-test-label">{r.label}</span>
                                                            {r.note ? (
                                                                <span className="sq-test-note">{r.note}</span>
                                                            ) : (
                                                                <span className="sq-test-classes">
                                                                    {r.per_class.map((c, j) => (
                                                                        <span key={j} className={`sq-test-class ${c.matched ? 'sq-test-class--ok' : 'sq-test-class--miss'}`}>
                                                                            {c.class_name} ({(c.best_overlap * 100).toFixed(0)}%)
                                                                        </span>
                                                                    ))}
                                                                </span>
                                                            )}
                                                        </li>
                                                    ))}
                                                </ol>
                                            )}
                                        </>
                                    )}
                                </div>
                            </div>

                            <div className="sq-builder-footer">
                                <button className="sq-btn-cancel" onClick={() => setEditing(false)}>Cancel</button>
                                <button className="sq-btn-test" onClick={handleTestOnImage} disabled={testing}>
                                    {testing ? 'Testing…' : 'Test on Image'}
                                </button>
                                <button className="sq-btn-save" onClick={handleSave}><Check size={15} /> Save Sequence</button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
