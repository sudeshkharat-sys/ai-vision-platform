import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Scissors, X, RefreshCw } from 'lucide-react';
import { SEG_MODEL_GROUPS, DEFAULT_SEG_MODEL } from '../constants/yoloModels';
// Reuse MainTrainingPanel's styling (mtp-* classes) — same visual language,
// no need for a near-duplicate stylesheet.
import './MainTrainingPanel.css';

import { API_URL } from '../config';
const POLL_INTERVAL = 3000;
const NO_WORKER_TICKS = 15; // ~45 s

const fmtEta = (sec) => {
    if (sec == null || sec <= 0) return null;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
};

const ChartTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="chart-tooltip">
            <p className="chart-tooltip-epoch">Epoch {label}</p>
            {payload.map(p => (
                <p key={p.dataKey} style={{ color: p.color }}>
                    {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
                </p>
            ))}
        </div>
    );
};

const MapChart = ({ history }) => {
    if (!history?.length) return null;
    const hasMap50 = history.some(h => h['mAP50'] != null || h['mAP50(B)'] != null);
    if (!hasMap50) return null;
    const data = history.map(h => ({ epoch: h.epoch, mAP50: h['mAP50'] ?? h['mAP50(B)'] ?? null }));
    return (
        <div className="chart-wrap">
            <p className="chart-title">Validation mAP50 (mask)</p>
            <ResponsiveContainer width="100%" height={160}>
                <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis dataKey="epoch" tick={{ fill: '#666666', fontSize: 10 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: '#666666', fontSize: 10 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 10, color: '#666666' }} />
                    <Line type="monotone" dataKey="mAP50" name="mAP50" stroke="#10b981" dot={false} strokeWidth={1.5} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

const EpochProgress = ({ meta }) => {
    if (!meta || meta.epoch == null) return null;
    const { epoch, total_epochs, eta_seconds } = meta;
    const pct = Math.round((epoch / total_epochs) * 100);
    const eta = fmtEta(eta_seconds);
    return (
        <div className="epoch-progress">
            <div className="epoch-progress-header">
                <span className="epoch-label">Epoch <strong>{epoch}</strong> / {total_epochs}</span>
                {eta && <span className="epoch-eta">⏱ ETA: {eta}</span>}
                <span className="epoch-pct">{pct}%</span>
            </div>
            <div className="epoch-bar">
                <div className="epoch-bar-fill mtp-bar-fill" style={{ width: `${pct}%`, background: 'linear-gradient(90deg,#10b981,#34d399)' }} />
            </div>
        </div>
    );
};

// ══════════════════════════════════════════════════════════════════
//  SegTrainingPanel — instance-segmentation training (YOLO-seg)
// ══════════════════════════════════════════════════════════════════
const SegTrainingPanel = ({ project, onClose }) => {
    const [segStatus, setSegStatus] = useState(null);
    const [statusLoading, setStatusLoading] = useState(true);
    const [selectedModel, setSelectedModel] = useState(DEFAULT_SEG_MODEL);
    const [epochs, setEpochs] = useState(100);
    const [imgsz, setImgsz] = useState(640);
    const [batch, setBatch] = useState(-1);
    const [preprocess, setPreprocess] = useState(true);
    const [launching, setLaunching] = useState(false);
    const [job, setJob] = useState(null);
    const [converting, setConverting] = useState(false);
    const [convertResult, setConvertResult] = useState(null);
    const [convertingBoxes, setConvertingBoxes] = useState(false);
    const [convertBoxesResult, setConvertBoxesResult] = useState(null);

    const pollRef = useRef(null);

    const convertBoxes = useCallback(async () => {
        setConvertingBoxes(true);
        setConvertBoxesResult(null);
        try {
            const res = await axios.patch(
                `${API_URL}/annotations/project/${project.id}/boxes-to-polygon`
            );
            setConvertBoxesResult(res.data);
        } catch (e) {
            setConvertBoxesResult({ error: e.response?.data?.detail || 'Conversion failed.' });
        } finally {
            setConvertingBoxes(false);
        }
    }, [project.id]);

    const convertPolygons = useCallback(async () => {
        setConverting(true);
        setConvertResult(null);
        try {
            const res = await axios.patch(
                `${API_URL}/annotations/project/${project.id}/polygons-to-segment`
            );
            setConvertResult(res.data);
        } catch (e) {
            setConvertResult({ error: e.response?.data?.detail || 'Conversion failed.' });
        } finally {
            setConverting(false);
        }
    }, [project.id]);

    const loadStatus = useCallback(() => {
        setStatusLoading(true);
        axios.get(`${API_URL}/pipeline/model-details/${project.id}`)
            .then(res => setSegStatus(res.data.seg || null))
            .catch(() => {})
            .finally(() => setStatusLoading(false));
    }, [project.id]);

    useEffect(() => {
        loadStatus();
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, [loadStatus]);

    const startPolling = useCallback((taskId) => {
        if (pollRef.current) clearInterval(pollRef.current);
        let pendingTicks = 0;

        pollRef.current = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/pipeline/task-status/${taskId}`);
                const { status, result, meta, error } = res.data;

                if (status === 'PENDING') {
                    pendingTicks++;
                    if (pendingTicks >= NO_WORKER_TICKS) {
                        clearInterval(pollRef.current);
                        axios.post(`${API_URL}/pipeline/cancel/${taskId}`).catch(() => {});
                        setJob(j => ({ ...j, status: 'NO_WORKER', error: 'No Celery worker detected after 45s.' }));
                    }
                    return;
                }
                if (status === 'STARTED') {
                    setJob(j => ({ ...j, status: 'STARTED', epochMeta: meta || j?.epochMeta }));
                    return;
                }
                if (status === 'SUCCESS') {
                    clearInterval(pollRef.current);
                    if (result?.error) {
                        setJob(j => ({ ...j, status: 'FAILURE', error: result.error }));
                    } else {
                        setJob(j => ({ ...j, status: 'SUCCESS', result, epochMeta: result?.history?.length
                            ? { ...j?.epochMeta, history: result.history, epoch: result.history.length }
                            : j?.epochMeta }));
                        loadStatus();
                    }
                    return;
                }
                if (status === 'FAILURE') {
                    clearInterval(pollRef.current);
                    setJob(j => ({ ...j, status: 'FAILURE', error: error || 'Unknown error' }));
                }
            } catch { /* ignore poll errors */ }
        }, POLL_INTERVAL);
    }, [loadStatus]);

    const handleTrain = async () => {
        setLaunching(true);
        try {
            const res = await axios.post(`${API_URL}/pipeline/train-seg/${project.id}`, {
                model_name: selectedModel, epochs, imgsz, preprocess, batch,
            });
            const taskId = res.data.task_id;
            setJob({ taskId, status: 'PENDING', epochMeta: null, result: null, error: null });
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: taskId, project_id: project.id, job_type: 'seg_training',
                result_meta: { modelName: selectedModel },
            }).catch(() => {});
            startPolling(taskId);
        } catch (err) {
            setJob({ status: 'FAILURE', error: err.response?.data?.detail || 'Failed to queue segmentation training.' });
        } finally {
            setLaunching(false);
        }
    };

    const running = job?.status === 'PENDING' || job?.status === 'STARTED';

    return (
        <div className="mtp-overlay" onClick={onClose}>
            <div className="mtp-panel" onClick={e => e.stopPropagation()}>
                <div className="mtp-header">
                    <div className="mtp-header-left">
                        <span className="mtp-header-icon"><Scissors size={20} /></span>
                        <div>
                            <h2 className="mtp-title">Train Segmentation Model</h2>
                            <p className="mtp-subtitle">{project.name}</p>
                        </div>
                    </div>
                    <button className="mtp-close" onClick={onClose}><X size={18} /></button>
                </div>

                <div className="mtp-body">
                    <section className="mtp-section">
                        <div className="mtp-section-header">
                            <span className="mtp-section-title">Segmentation Model</span>
                            <button className="mtp-refresh" onClick={loadStatus} title="Refresh"><RefreshCw size={15} /></button>
                        </div>
                        {statusLoading ? (
                            <div className="mtp-loading"><div className="mtp-spinner" /><span>Loading…</span></div>
                        ) : (
                            <div className="mtp-model-status-row">
                                <div className={`mtp-model-chip ${segStatus?.exists ? 'mtp-model-chip--ok' : 'mtp-model-chip--none'}`}>
                                    {segStatus?.exists ? '✅' : '○'} seg_best.pt {segStatus?.exists ? `(${segStatus.file_size_mb} MB)` : '(not trained yet)'}
                                </div>
                            </div>
                        )}
                        <div className="mtp-warning" style={{ marginTop: 8 }}>
                            Draw at least one annotation with the Segment tool (mask outline, not a box or precision polyline) before training — bbox-only images are skipped.
                        </div>
                        <div className="mtp-warning" style={{ marginTop: 8 }}>
                            Still have plain box annotations across this project? Turn every
                            box into a 4-corner polygon in one click — then drag corners onto
                            the real outline where needed before converting to segment masks.
                            <div style={{ marginTop: 8 }}>
                                <button
                                    className="mtp-refresh"
                                    style={{ width: 'auto', padding: '6px 12px' }}
                                    disabled={convertingBoxes}
                                    onClick={convertBoxes}
                                >
                                    {convertingBoxes ? 'Converting…' : 'Convert all boxes → polygons'}
                                </button>
                            </div>
                            {convertBoxesResult && (
                                <p style={{ marginTop: 6 }}>
                                    {convertBoxesResult.error
                                        ? `❌ ${convertBoxesResult.error}`
                                        : `✅ Converted ${convertBoxesResult.converted} of ${convertBoxesResult.total_annotations} annotation(s) to polygons.`}
                                </p>
                            )}
                        </div>
                        <div className="mtp-warning" style={{ marginTop: 8 }}>
                            Already traced plate/character outlines with the Polyline tool
                            (annotation type "polygon") in this project? Those are real
                            outlines, not boxes — re-tag them as Segment masks below instead
                            of redrawing everything.
                            <div style={{ marginTop: 8 }}>
                                <button
                                    className="mtp-refresh"
                                    style={{ width: 'auto', padding: '6px 12px' }}
                                    disabled={converting}
                                    onClick={convertPolygons}
                                >
                                    {converting ? 'Converting…' : 'Convert existing polygons → segment masks'}
                                </button>
                            </div>
                            {convertResult && (
                                <p style={{ marginTop: 6 }}>
                                    {convertResult.error
                                        ? `❌ ${convertResult.error}`
                                        : `✅ Converted ${convertResult.converted} of ${convertResult.total_polygons} polygon annotation(s) to segment masks.`}
                                </p>
                            )}
                        </div>
                    </section>

                    <section className="mtp-section">
                        <p className="mtp-section-title">Training Config</p>

                        <div className="mtp-model-row">
                            <label className="mtp-model-label">Starting Weights (YOLO-seg)</label>
                            <select
                                className="mtp-model-select"
                                value={selectedModel}
                                onChange={e => setSelectedModel(e.target.value)}
                                disabled={running}
                            >
                                {SEG_MODEL_GROUPS.map(group => (
                                    <optgroup key={group.family} label={`${group.family}${group.note ? ` (${group.note})` : ''}`}>
                                        {group.models.map(m => (
                                            <option key={m.value} value={m.value}>{m.label}  [{m.params}]</option>
                                        ))}
                                    </optgroup>
                                ))}
                            </select>
                        </div>

                        <div className="mtp-model-row">
                            <label className="mtp-model-label">
                                Epochs
                                <span className="mtp-model-hint">More = better mask accuracy, longer training</span>
                            </label>
                            <div className="mtp-epochs-row">
                                <input
                                    type="range" min="10" max="500" step="10"
                                    value={epochs}
                                    onChange={e => setEpochs(Number(e.target.value))}
                                    className="mtp-epochs-slider"
                                    disabled={running}
                                />
                                <span className="mtp-epochs-val">{epochs}</span>
                            </div>
                        </div>

                        <div className="mtp-model-row">
                            <label className="mtp-model-label">Image Size</label>
                            <select
                                className="mtp-model-select mtp-model-select--sm"
                                value={imgsz}
                                onChange={e => setImgsz(Number(e.target.value))}
                                disabled={running}
                            >
                                {[320, 416, 512, 640, 768, 1024].map(s => (
                                    <option key={s} value={s}>{s} × {s}{s === 640 ? ' (recommended)' : ''}</option>
                                ))}
                            </select>
                        </div>

                        <div className="mtp-model-row">
                            <label className="mtp-model-label">Batch Size</label>
                            <select
                                className="mtp-model-select mtp-model-select--sm"
                                value={batch}
                                onChange={e => setBatch(Number(e.target.value))}
                                disabled={running}
                            >
                                <option value={-1}>Auto (recommended)</option>
                                {[2, 4, 8, 16, 32].map(b => <option key={b} value={b}>{b}</option>)}
                            </select>
                        </div>

                        <div className="mtp-toggle-row" style={{ marginTop: 12 }}>
                            <label className="mtp-toggle-label">
                                <input
                                    type="checkbox"
                                    className="mtp-toggle-check"
                                    checked={preprocess}
                                    onChange={e => setPreprocess(e.target.checked)}
                                    disabled={running}
                                />
                                <span className="mtp-toggle-slider" />
                                <span className="mtp-toggle-text">CLAHE contrast preprocessing</span>
                            </label>
                        </div>
                    </section>

                    {job && (
                        <section className="mtp-section">
                            <p className="mtp-section-title">Job Status</p>
                            {job.status === 'STARTED' && <EpochProgress meta={job.epochMeta} />}
                            {job.epochMeta?.history?.length > 0 && <MapChart history={job.epochMeta.history} />}
                            {job.status === 'SUCCESS' && (
                                <div className="mtp-info">✅ Segmentation training complete — {job.result?.model_path}</div>
                            )}
                            {(job.status === 'FAILURE' || job.status === 'NO_WORKER') && (
                                <div className="mtp-error-text">❌ {job.error}</div>
                            )}
                            {job.status === 'PENDING' && (
                                <div className="mtp-info">⏳ Waiting for a Celery worker…</div>
                            )}
                        </section>
                    )}
                </div>

                <div className="mtp-footer">
                    <button
                        className="mtp-train-btn"
                        onClick={handleTrain}
                        disabled={launching || running}
                    >
                        {launching ? 'Queuing…' : running ? 'Training…' : <><Scissors size={16} /> Start Segmentation Training</>}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SegTrainingPanel;
