import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Scissors, X, RefreshCw, Download, Inbox, Square, Copy } from 'lucide-react';
import { SEG_MODEL_GROUPS, DEFAULT_SEG_MODEL } from '../constants/yoloModels';
// Reuse MainTrainingPanel's styling (mtp-* classes) — same visual language,
// no need for a near-duplicate stylesheet.
import './MainTrainingPanel.css';

import { API_URL } from '../config';
const POLL_INTERVAL = 3000;
const MAX_PARALLEL = 2;
const NO_WORKER_TICKS = 15; // ~45 s

// ── Split preview (mirrors backend _split_images logic) ──────────────
function computeSplitPreview(n) {
    if (n < 1) return null;
    if (n < 5)  return { train: n, val: n, test: 0, mirror: true };
    if (n < 10) {
        const train = Math.max(1, Math.round(n * 0.8));
        return { train, val: n - train, test: 0, mirror: false };
    }
    const train = Math.max(1, Math.round(n * 0.8));
    const val   = Math.max(1, Math.round(n * 0.15));
    const test  = n - train - val > 0 ? n - train - val : 0;
    return { train, val, test, mirror: false };
}

// ── Helpers ───────────────────────────────────────────
const makeJob = (taskId, modelName, modelType) => ({
    id: Date.now(),
    taskId,
    modelName,
    modelType,
    status: 'PENDING',
    logs: [`📋  Task ID: ${taskId}`, '⏳  Waiting for worker…'],
    epochMeta: null,
    result: null,
    error: null,
    startedAt: new Date(),
});

const STATUS_LABEL = {
    QUEUED:    { label: 'Queued',    cls: 'badge--queued'  },
    PENDING:   { label: 'Pending',   cls: 'badge--pending' },
    STARTED:   { label: 'Running',   cls: 'badge--running' },
    SUCCESS:   { label: 'Done',      cls: 'badge--done'    },
    FAILURE:   { label: 'Failed',    cls: 'badge--fail'    },
    REVOKED:   { label: 'Stopped',   cls: 'badge--fail'    },
    NO_WORKER: { label: 'No Worker', cls: 'badge--fail'    },
};

const fmtTime = (date) =>
    new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

const fmtEta = (sec) => {
    if (sec == null || sec <= 0) return null;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
};

// ── Tiny chart tooltip ─────────────────────────────────
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

// ── Loss chart (box/seg/cls/dfl) ───────────────────────
const LossChart = ({ history }) => {
    if (!history?.length) return null;
    const hasBox = history.some(h => h.box_loss != null);
    const hasSeg = history.some(h => h.seg_loss != null);
    const hasCls = history.some(h => h.cls_loss != null);
    const hasDfl = history.some(h => h.dfl_loss != null);
    if (!hasBox && !hasSeg && !hasCls && !hasDfl) return null;
    return (
        <div className="chart-wrap">
            <p className="chart-title">Training Loss</p>
            <ResponsiveContainer width="100%" height={160}>
                <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis dataKey="epoch" tick={{ fill: '#666666', fontSize: 10 }} label={{ value: 'Epoch', position: 'insideBottom', fill: '#666666', fontSize: 10, offset: -1 }} />
                    <YAxis tick={{ fill: '#666666', fontSize: 10 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 10, color: '#666666' }} />
                    {hasBox && <Line type="monotone" dataKey="box_loss" name="Box" stroke="#dc143c" dot={false} strokeWidth={1.5} />}
                    {hasSeg && <Line type="monotone" dataKey="seg_loss" name="Seg (mask)" stroke="#10b981" dot={false} strokeWidth={1.5} />}
                    {hasCls && <Line type="monotone" dataKey="cls_loss" name="Cls" stroke="#f59e0b" dot={false} strokeWidth={1.5} />}
                    {hasDfl && <Line type="monotone" dataKey="dfl_loss" name="DFL" stroke="#ec4899" dot={false} strokeWidth={1.5} />}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

// ── mAP chart (box + mask) ─────────────────────────────
const MapChart = ({ history }) => {
    if (!history?.length) return null;
    const hasBox50  = history.some(h => h['mAP50'] != null);
    const hasBox95  = history.some(h => h['mAP50-95'] != null);
    const hasMask50 = history.some(h => h['mAP50(M)'] != null);
    const hasMask95 = history.some(h => h['mAP50-95(M)'] != null);
    if (!hasBox50 && !hasBox95 && !hasMask50 && !hasMask95) return null;

    const data = history.map(h => ({
        epoch: h.epoch,
        mAP50: h['mAP50'] ?? null,
        'mAP50-95': h['mAP50-95'] ?? null,
        'mAP50(M)': h['mAP50(M)'] ?? null,
        'mAP50-95(M)': h['mAP50-95(M)'] ?? null,
    }));

    return (
        <div className="chart-wrap">
            <p className="chart-title">Validation mAP (box + mask)</p>
            <ResponsiveContainer width="100%" height={160}>
                <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis dataKey="epoch" tick={{ fill: '#666666', fontSize: 10 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: '#666666', fontSize: 10 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 10, color: '#666666' }} />
                    {hasBox50  && <Line type="monotone" dataKey="mAP50" name="Box mAP50" stroke="#4ade80" dot={false} strokeWidth={1.5} />}
                    {hasBox95  && <Line type="monotone" dataKey="mAP50-95" name="Box mAP50-95" stroke="#38bdf8" dot={false} strokeWidth={1.5} />}
                    {hasMask50 && <Line type="monotone" dataKey="mAP50(M)" name="Mask mAP50" stroke="#10b981" dot={false} strokeWidth={1.5} />}
                    {hasMask95 && <Line type="monotone" dataKey="mAP50-95(M)" name="Mask mAP50-95" stroke="#a78bfa" dot={false} strokeWidth={1.5} />}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

// ── PreprocessingProgress ──────────────────────────────
const PreprocessingProgress = ({ meta }) => {
    if (!meta || meta.phase !== 'preprocessing') return null;
    const { current = 0, total = 0, split = '', pct = 0 } = meta;
    const splitLabel = { train: 'train', val: 'validation', test: 'test' }[split] || split;
    return (
        <div className="epoch-progress">
            <div className="epoch-progress-header">
                <span className="epoch-label">
                    ⚡ Preprocessing
                    {splitLabel && <span style={{ color: '#666666', fontWeight: 400 }}> — {splitLabel} set</span>}
                </span>
                <span className="epoch-eta" style={{ color: '#d97706' }}>{current} / {total} images</span>
                <span className="epoch-pct">{pct}%</span>
            </div>
            <div className="epoch-bar">
                <div className="epoch-bar-fill mtp-bar-fill" style={{ width: `${pct}%`, background: 'linear-gradient(90deg,#dc143c,#f87171)' }} />
            </div>
            <p style={{ fontSize: 10, color: '#666666', marginTop: 4 }}>
                Applying CLAHE contrast enhancement to training images…
            </p>
        </div>
    );
};

// ── EpochProgress ──────────────────────────────────────
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

// ══════════════════════════════════════════════════════
//  SegTrainingPanel — instance-segmentation training (YOLO-seg)
// ══════════════════════════════════════════════════════
const SegTrainingPanel = ({ project, onClose }) => {
    const [stats, setStats]             = useState(null);
    const [statsLoading, setStatsLoading] = useState(true);
    const [segSeedStatus, setSegSeedStatus] = useState(null);
    const [segMainStatus, setSegMainStatus] = useState(null);
    const [segLegacyStatus, setSegLegacyStatus] = useState(null);
    const [jobs, setJobs]               = useState([]);
    const [activeJobId, setActiveJobId] = useState(null);
    const [launching, setLaunching]     = useState(false);
    const [view, setView]               = useState('detail');
    const [selectedModel, setSelectedModel] = useState(DEFAULT_SEG_MODEL);
    const [modelType, setModelType]     = useState('main'); // 'seed' | 'main'
    const [epochs, setEpochs]           = useState(100);
    const [imgsz, setImgsz]             = useState(640);
    const [batch, setBatch]             = useState(-1);
    const [preprocess, setPreprocess]   = useState(true);
    const [augFliplr, setAugFliplr]     = useState(true);
    const [augFlipud, setAugFlipud]     = useState(true);
    const [augMosaic, setAugMosaic]     = useState(true);
    const [augHsvV, setAugHsvV]         = useState(0.4);
    const [augHsvH, setAugHsvH]         = useState(0.015);
    const [augHsvS, setAugHsvS]         = useState(0.3);
    const [augDegrees, setAugDegrees]   = useState(10);
    const [augTranslate, setAugTranslate] = useState(0.1);
    const [augScale, setAugScale]       = useState(0.4);
    const [augMixup, setAugMixup]       = useState(0.0);
    const [augCopyPaste, setAugCopyPaste] = useState(0.05);
    const [showAugSettings, setShowAugSettings] = useState(false);
    const [converting, setConverting]   = useState(false);
    const [convertResult, setConvertResult] = useState(null);
    const [convertingBoxes, setConvertingBoxes] = useState(false);
    const [convertBoxesResult, setConvertBoxesResult] = useState(null);

    const logsEndRef     = useRef(null);
    const pollRef        = useRef({});
    const jobsRef        = useRef(jobs);
    const queueRef       = useRef([]);
    const autoRemoveRef  = useRef({});

    useEffect(() => { jobsRef.current = jobs; }, [jobs]);

    // Auto-remove failed/stopped/no-worker jobs after 4 s
    useEffect(() => {
        const TERMINAL = ['FAILURE', 'REVOKED', 'NO_WORKER'];
        jobs.forEach(j => {
            if (TERMINAL.includes(j.status) && !autoRemoveRef.current[j.id]) {
                autoRemoveRef.current[j.id] = setTimeout(() => {
                    delete autoRemoveRef.current[j.id];
                    setJobs(prev => prev.filter(jj => jj.id !== j.id));
                    if (j.taskId) axios.delete(`${API_URL}/pipeline/jobs/${j.taskId}`).catch(() => {});
                }, 4000);
            }
        });
        Object.keys(autoRemoveRef.current).forEach(id => {
            if (!jobs.find(j => String(j.id) === id)) {
                clearTimeout(autoRemoveRef.current[id]);
                delete autoRemoveRef.current[id];
            }
        });
    }, [jobs]);

    // ── Stats + model status ───────────────────────────
    const loadStatus = useCallback(() => {
        setStatsLoading(true);
        Promise.all([
            axios.get(`${API_URL}/pipeline/training-stats/${project.id}`),
            axios.get(`${API_URL}/pipeline/model-details/${project.id}`),
        ])
            .then(([statsRes, detailsRes]) => {
                setStats(statsRes.data);
                setSegSeedStatus(detailsRes.data.seg_seed || null);
                setSegMainStatus(detailsRes.data.seg_main || null);
                setSegLegacyStatus(detailsRes.data.seg || null);
            })
            .catch(() => {})
            .finally(() => setStatsLoading(false));
    }, [project.id]);

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

    // ── Load persisted jobs from DB on mount ──────────
    const loadPersistedJobs = useCallback((resumePollingFn) => {
        axios.get(`${API_URL}/pipeline/jobs/${project.id}?job_type=seg_training`)
            .then(async res => {
                if (!res.data.length) return;
                const DB_STATUS = { pending: 'PENDING', started: 'STARTED', success: 'SUCCESS', failure: 'FAILURE', revoked: 'REVOKED' };
                const loaded = res.data.map(j => ({
                    id: j.id,
                    taskId: j.id,
                    modelName: j.result_meta?.modelName || DEFAULT_SEG_MODEL,
                    modelType: j.result_meta?.modelType || 'main',
                    status: DB_STATUS[j.status] || 'PENDING',
                    logs: j.result_meta?.logs || [`📋  Task ID: ${j.id}`],
                    epochMeta: j.result_meta?.epochMeta || null,
                    result:    j.result_meta?.result   || null,
                    error:     j.result_meta?.error    || null,
                    startedAt: new Date(j.created_at),
                }));

                const reconciled = await Promise.all(loaded.map(async (job) => {
                    if (job.status !== 'PENDING' && job.status !== 'STARTED') return job;

                    try {
                        const sr = await axios.get(`${API_URL}/pipeline/task-status/${job.taskId}`);
                        const { status: celery, result, error } = sr.data;

                        if (celery === 'SUCCESS') {
                            const newLogs = [...job.logs, '✅  Segmentation training complete! (completed while panel was closed)'];
                            const finalMeta = result?.history?.length
                                ? { ...job.epochMeta, history: result.history, epoch: result.history.length }
                                : job.epochMeta;
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'success',
                                result_meta: { logs: newLogs, epochMeta: finalMeta, result, modelName: job.modelName, modelType: job.modelType },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            loadStatus();
                            return { ...job, status: 'SUCCESS', result, logs: newLogs, epochMeta: finalMeta };

                        } else if (celery === 'FAILURE') {
                            const newLogs = [...job.logs, `❌  Failed: ${error || 'Unknown error'}`];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { logs: newLogs, error, modelName: job.modelName, modelType: job.modelType },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...job, status: 'FAILURE', error, logs: newLogs };

                        } else if (celery === 'PENDING' && job.status === 'STARTED') {
                            const newLogs = [...job.logs,
                                '⚠️  Task result expired (panel was closed before it finished).',
                                '    The training run may have completed — check for a saved model.',
                            ];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { logs: newLogs, modelName: job.modelName, modelType: job.modelType },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...job, status: 'FAILURE', logs: newLogs };

                        } else {
                            resumePollingFn(job.taskId);
                            return job;
                        }
                    } catch {
                        resumePollingFn(job.taskId);
                        return job;
                    }
                }));

                setJobs(reconciled);
                setActiveJobId(reconciled[reconciled.length - 1].id);
            })
            .catch(() => {});
    }, [project.id, loadStatus]);

    useEffect(() => {
        loadStatus();
        loadPersistedJobs(startPolling);
        const polls = pollRef.current;
        return () => { Object.values(polls).forEach(clearInterval); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [loadStatus, loadPersistedJobs]);

    // auto-scroll log
    useEffect(() => {
        if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [jobs, activeJobId]);

    const runningCount = () =>
        jobsRef.current.filter(j => j.status === 'PENDING' || j.status === 'STARTED').length;

    // ── Polling ───────────────────────────────────────
    const startPolling = useCallback((taskId) => {
        if (pollRef.current[taskId]) clearInterval(pollRef.current[taskId]);
        let pendingTicks = 0;
        let startedPersisted = false;

        pollRef.current[taskId] = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/pipeline/task-status/${taskId}`);
                const { status, result, meta, error } = res.data;

                setJobs(prev => prev.map(j => {
                    if (j.taskId !== taskId) return j;
                    let newLogs = [...j.logs];

                    if (status === 'PENDING') {
                        pendingTicks++;
                        if (pendingTicks >= NO_WORKER_TICKS) {
                            clearInterval(pollRef.current[taskId]);
                            delete pollRef.current[taskId];
                            axios.post(`${API_URL}/pipeline/cancel/${taskId}`).catch(() => {});
                            newLogs = [...newLogs,
                                '❌  No Celery worker detected after 45s.',
                                '👉  Start a worker — see instructions below.'];
                            setTimeout(() => dispatchQueued(), 100);
                            return { ...j, status: 'NO_WORKER', logs: newLogs };
                        }
                        const msg = `⏳  Waiting for worker… (${pendingTicks * 3}s)`;
                        const last = newLogs[newLogs.length - 1];
                        newLogs = last?.startsWith('⏳')
                            ? [...newLogs.slice(0, -1), msg]
                            : [...newLogs, msg];
                        return { ...j, status: 'PENDING', logs: newLogs };
                    }

                    if (status === 'STARTED') {
                        pendingTicks = 0;
                        if (!startedPersisted) {
                            startedPersisted = true;
                            axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, { status: 'started' }).catch(() => {});
                        }
                        const phase = meta?.phase;
                        if (phase === 'preprocessing') {
                            const pct = meta.pct ?? 0;
                            const msg = `⚡  Preprocessing images… ${pct}% (${meta.current ?? 0}/${meta.total ?? 0})`;
                            const last = newLogs[newLogs.length - 1];
                            newLogs = last?.startsWith('⚡')
                                ? [...newLogs.slice(0, -1), msg]
                                : [...newLogs, '⚡  Starting CLAHE preprocessing…', msg];
                        } else {
                            const last = newLogs[newLogs.length - 1];
                            const wasPreprocessing = last?.startsWith('⚡');
                            if (wasPreprocessing)
                                newLogs = [...newLogs, '✔  Preprocessing done — starting training…'];
                            else if (last !== '⚙️  Segmentation training in progress…')
                                newLogs = [...newLogs, '⚙️  Segmentation training in progress…'];
                        }
                        return { ...j, status: 'STARTED', logs: newLogs, epochMeta: meta || j.epochMeta };
                    }

                    if (status === 'SUCCESS') {
                        clearInterval(pollRef.current[taskId]);
                        delete pollRef.current[taskId];
                        if (result?.error) {
                            newLogs = [...newLogs, `❌  Failed: ${result.error}`];
                            setTimeout(() => dispatchQueued(), 100);
                            axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                                status: 'failure',
                                result_meta: { logs: newLogs, error: result.error, modelName: j.modelName, modelType: j.modelType },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...j, status: 'FAILURE', error: result.error, logs: newLogs };
                        }
                        newLogs = [...newLogs, '✅  Segmentation training complete!',
                            result?.model_path ? `📦  Model: ${result.model_path}` : ''].filter(Boolean);
                        loadStatus();
                        setTimeout(() => dispatchQueued(), 100);
                        const finalMeta = result?.history?.length
                            ? { ...j.epochMeta, history: result.history, epoch: result.history.length }
                            : j.epochMeta;
                        axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                            status: 'success',
                            result_meta: {
                                logs: newLogs, epochMeta: finalMeta, result,
                                modelName: j.modelName, modelType: j.modelType,
                                startedAt: j.startedAt instanceof Date ? j.startedAt.toISOString() : j.startedAt,
                            },
                            finished_at: new Date().toISOString(),
                        }).catch(() => {});
                        return { ...j, status: 'SUCCESS', result, logs: newLogs, epochMeta: finalMeta };
                    }

                    if (status === 'FAILURE') {
                        clearInterval(pollRef.current[taskId]);
                        delete pollRef.current[taskId];
                        newLogs = [...newLogs, `❌  Failed: ${error || 'Unknown error'}`];
                        setTimeout(() => dispatchQueued(), 100);
                        axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                            status: 'failure',
                            result_meta: {
                                logs: newLogs, error: error || 'Unknown error',
                                modelName: j.modelName, modelType: j.modelType,
                                startedAt: j.startedAt instanceof Date ? j.startedAt.toISOString() : j.startedAt,
                            },
                            finished_at: new Date().toISOString(),
                        }).catch(() => {});
                        return { ...j, status: 'FAILURE', error: error || 'Unknown error', logs: newLogs };
                    }

                    return j;
                }));
            } catch { /* ignore poll errors */ }
        }, POLL_INTERVAL);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [loadStatus]);

    // ── Dispatch queued job ────────────────────────────
    const dispatchQueued = useCallback(async () => {
        if (runningCount() >= MAX_PARALLEL) return;
        const next = queueRef.current.shift();
        if (!next) return;
        try {
            const res = await axios.post(`${API_URL}/pipeline/train-seg/${next.projectId}`, {
                model_name: next.modelName, model_type: next.modelType, epochs: next.epochs,
                imgsz: next.imgsz, preprocess: next.preprocess, batch: next.batch,
                aug_fliplr: next.augFliplr ? 0.5 : 0.0,
                aug_flipud: next.augFlipud ? 0.1 : 0.0,
                aug_mosaic: next.augMosaic ? 0.5 : 0.0,
                aug_hsv_v: next.augHsvV, aug_hsv_h: next.augHsvH, aug_hsv_s: next.augHsvS,
                aug_degrees: next.augDegrees, aug_translate: next.augTranslate, aug_scale: next.augScale,
                aug_mixup: next.augMixup, aug_copy_paste: next.augCopyPaste,
            });
            const taskId = res.data.task_id;
            const logs = [`📋  Task ID: ${taskId}`, '⏳  Waiting for worker…'];
            setJobs(prev => prev.map(j =>
                j.id === next.jobId
                    ? { ...j, taskId, status: 'PENDING', logs }
                    : j
            ));
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: taskId, project_id: next.projectId, job_type: 'seg_training',
                result_meta: { logs, startedAt: new Date().toISOString(), modelName: next.modelName, modelType: next.modelType },
            }).catch(() => {});
            startPolling(taskId);
        } catch {
            setJobs(prev => prev.map(j =>
                j.id === next.jobId
                    ? { ...j, status: 'FAILURE', logs: [...j.logs, '❌  Failed to queue job.'] }
                    : j
            ));
        }
    }, [startPolling]);

    // ── Remove job from UI + DB ───────────────────────
    const removeJob = useCallback((job) => {
        setJobs(prev => prev.filter(j => j.id !== job.id));
        if (job.taskId) axios.delete(`${API_URL}/pipeline/jobs/${job.taskId}`).catch(() => {});
    }, []);

    const clearFailedJobs = useCallback(() => {
        const failed = jobsRef.current.filter(j => ['FAILURE','REVOKED','NO_WORKER'].includes(j.status));
        failed.forEach(j => { if (j.taskId) axios.delete(`${API_URL}/pipeline/jobs/${j.taskId}`).catch(() => {}); });
        setJobs(prev => prev.filter(j => !['FAILURE','REVOKED','NO_WORKER'].includes(j.status)));
    }, []);

    // ── Force stop all running/queued jobs ────────────────────────
    const handleForceStopAll = async () => {
        try {
            await axios.post(`${API_URL}/pipeline/force-stop-all/${project.id}`);
        } catch { /* ignore network errors */ }
        Object.values(pollRef.current).forEach(clearInterval);
        pollRef.current = {};
        setJobs(prev => prev.map(j =>
            (j.status === 'PENDING' || j.status === 'STARTED' || j.status === 'QUEUED')
                ? { ...j, status: 'REVOKED', logs: [...j.logs, '🛑  Force stopped by user.'] }
                : j
        ));
        queueRef.current = [];
    };

    // ── Launch training ────────────────────────────────
    const handleTrain = async () => {
        setLaunching(true);
        setView('jobs');

        if (runningCount() >= MAX_PARALLEL) {
            const placeholder = {
                id: Date.now(), taskId: null, status: 'QUEUED',
                modelName: selectedModel, modelType,
                logs: ['📋  Job queued — waiting for a free slot…'],
                epochMeta: null, result: null, error: null, startedAt: new Date(),
            };
            queueRef.current.push({
                jobId: placeholder.id, projectId: project.id, modelName: selectedModel, modelType,
                epochs, preprocess, imgsz, batch,
                augFliplr, augFlipud, augMosaic, augHsvV, augHsvH, augHsvS,
                augDegrees, augTranslate, augScale, augMixup, augCopyPaste,
            });
            setJobs(prev => [...prev, placeholder]);
            setActiveJobId(placeholder.id);
            setLaunching(false);
            return;
        }

        try {
            const res = await axios.post(`${API_URL}/pipeline/train-seg/${project.id}`, {
                model_name: selectedModel, model_type: modelType, epochs, imgsz, preprocess, batch,
                aug_fliplr: augFliplr ? 0.5 : 0.0,
                aug_flipud: augFlipud ? 0.1 : 0.0,
                aug_mosaic: augMosaic ? 0.5 : 0.0,
                aug_hsv_v: augHsvV, aug_hsv_h: augHsvH, aug_hsv_s: augHsvS,
                aug_degrees: augDegrees, aug_translate: augTranslate, aug_scale: augScale,
                aug_mixup: augMixup, aug_copy_paste: augCopyPaste,
            });
            const taskId = res.data.task_id;
            const job = makeJob(taskId, selectedModel, modelType);
            setJobs(prev => [...prev, job]);
            setActiveJobId(job.id);
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: taskId, project_id: project.id, job_type: 'seg_training',
                result_meta: { logs: job.logs, startedAt: job.startedAt.toISOString(), modelName: selectedModel, modelType },
            }).catch(() => {});
            startPolling(taskId);
        } catch (err) {
            const failJob = {
                id: Date.now(), taskId: null, status: 'FAILURE',
                modelName: selectedModel, modelType,
                logs: ['❌  Failed to queue segmentation training job.'],
                epochMeta: null, result: null, error: err.response?.data?.detail || 'Failed to queue', startedAt: new Date(),
            };
            setJobs(prev => [...prev, failJob]);
            setActiveJobId(failJob.id);
        } finally {
            setLaunching(false);
        }
    };

    // ── Derived ────────────────────────────────────────
    const activeJob  = jobs.find(j => j.id === activeJobId) || jobs[jobs.length - 1] || null;
    const anyRunning = jobs.some(j => j.status === 'PENDING' || j.status === 'STARTED');
    const readyToTrain = stats?.ready_to_train;

    const btnLabel = () => {
        if (launching) return 'Queuing…';
        if (runningCount() >= MAX_PARALLEL) return <><Download size={16} /> Queue Job</>;
        return <><Scissors size={16} /> Train {modelType === 'seed' ? 'Seed' : 'Main'} Segmentation Model</>;
    };

    return (
        <div className="mtp-overlay" onClick={onClose}>
            <div className="mtp-panel" onClick={e => e.stopPropagation()}>

                {/* ── Header ── */}
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

                {/* ── Tabs ── */}
                <div className="mtp-tabs">
                    <button className={`mtp-tab ${view === 'detail' ? 'mtp-tab--active' : ''}`} onClick={() => setView('detail')}>
                        Dataset &amp; Config
                    </button>
                    <button className={`mtp-tab ${view === 'jobs' ? 'mtp-tab--active' : ''}`} onClick={() => setView('jobs')}>
                        Jobs
                        {jobs.length > 0 && <span className="mtp-tab-badge">{jobs.length}</span>}
                        {anyRunning && <span className="mtp-tab-dot" />}
                    </button>
                </div>

                <div className="mtp-body">

                    {/* ═══════════ DETAIL VIEW ═══════════ */}
                    {view === 'detail' && (
                        <>
                            <section className="mtp-section">
                                <div className="mtp-section-header">
                                    <span className="mtp-section-title">Dataset Overview</span>
                                    <button className="mtp-refresh" onClick={loadStatus} title="Refresh"><RefreshCw size={15} /></button>
                                </div>
                                {statsLoading ? (
                                    <div className="mtp-loading"><div className="mtp-spinner" /><span>Loading…</span></div>
                                ) : stats ? (
                                    <>
                                        <div className="mtp-stat-cards">
                                            <div className="mtp-stat-card"><span className="mtp-stat-value">{stats.total_images}</span><span className="mtp-stat-label">Total Images</span></div>
                                            <div className="mtp-stat-card mtp-stat-card--green"><span className="mtp-stat-value">{stats.annotated_images}</span><span className="mtp-stat-label">Annotated</span></div>
                                            <div className="mtp-stat-card mtp-stat-card--yellow"><span className="mtp-stat-value">{stats.pending_images}</span><span className="mtp-stat-label">Pending</span></div>
                                            <div className="mtp-stat-card mtp-stat-card--indigo"><span className="mtp-stat-value">{stats.total_annotations}</span><span className="mtp-stat-label">Annotations</span></div>
                                        </div>
                                        <div className="mtp-progress-wrap">
                                            <div className="mtp-progress-label">
                                                <span>Annotation Progress</span>
                                                <span>{stats.total_images > 0 ? Math.round((stats.annotated_images / stats.total_images) * 100) : 0}%</span>
                                            </div>
                                            <div className="mtp-progress-bar"><div className="mtp-progress-fill" style={{ width: stats.total_images > 0 ? `${(stats.annotated_images / stats.total_images) * 100}%` : '0%' }} /></div>
                                        </div>

                                        <div className="mtp-model-status-row">
                                            <div className={`mtp-model-chip ${segSeedStatus?.exists ? 'mtp-model-chip--ok' : 'mtp-model-chip--none'}`}>
                                                {segSeedStatus?.exists ? '✅' : '○'} Seed {segSeedStatus?.exists ? `(${segSeedStatus.file_size_mb} MB)` : '(not trained yet)'}
                                            </div>
                                            <div className={`mtp-model-chip ${segMainStatus?.exists ? 'mtp-model-chip--ok' : 'mtp-model-chip--none'}`}>
                                                {segMainStatus?.exists ? '✅' : '○'} Main {segMainStatus?.exists ? `(${segMainStatus.file_size_mb} MB)` : '(not trained yet)'}
                                            </div>
                                            {segLegacyStatus?.exists && (
                                                <div className="mtp-model-chip mtp-model-chip--ok" title="Trained before the seed/main split existed — still used as a fallback">
                                                    ✅ Legacy seg_best.pt ({segLegacyStatus.file_size_mb} MB)
                                                </div>
                                            )}
                                        </div>

                                        {!readyToTrain && <div className="mtp-warning">⚠️ Annotate at least 1 image before training.</div>}
                                        {stats.pending_images > 0 && readyToTrain && (
                                            <div className="mtp-info">{stats.pending_images} unannotated image{stats.pending_images !== 1 ? 's' : ''} — more annotations improve accuracy.</div>
                                        )}

                                        {/* ── Split preview ── */}
                                        {(() => {
                                            const sp = computeSplitPreview(stats.annotated_images);
                                            if (!sp) return null;
                                            return (
                                                <div className="mtp-split-preview">
                                                    <span className="mtp-split-preview-label">Expected split</span>
                                                    <div className="mtp-split-badges">
                                                        <span className="mtp-split-badge mtp-split-badge--train">Train&nbsp;{sp.train}</span>
                                                        <span className="mtp-split-badge mtp-split-badge--val">Val&nbsp;{sp.val}{sp.mirror ? ' (mirrors train)' : ''}</span>
                                                        {sp.test > 0 && <span className="mtp-split-badge mtp-split-badge--test">Test&nbsp;{sp.test}</span>}
                                                        <span className="mtp-split-badge mtp-split-badge--total">Total&nbsp;{stats.annotated_images}</span>
                                                    </div>
                                                </div>
                                            );
                                        })()}

                                        <div className="mtp-warning" style={{ marginTop: 8 }}>
                                            No segmentation model yet? Train <b>Seed</b> first on a small
                                            hand-annotated batch (20–50 images) so Auto-Annotate can propose
                                            masks for the rest — then train <b>Main</b> once more images are
                                            labeled. Auto-Annotate always prefers Main over Seed when both exist.
                                        </div>
                                        <div className="mtp-warning" style={{ marginTop: 8 }}>
                                            Draw at least one annotation with the Segment tool (mask outline, not a box or precision polyline) before training — bbox-only images are skipped.
                                        </div>
                                    </>
                                ) : <p className="mtp-error-text">Failed to load stats.</p>}
                            </section>

                            <section className="mtp-section">
                                <p className="mtp-section-title">Convert Existing Annotations</p>
                                <div className="mtp-warning">
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

                                <div className="mtp-tabs" style={{ marginBottom: 12 }}>
                                    <button
                                        type="button"
                                        className={`mtp-tab ${modelType === 'seed' ? 'mtp-tab--active' : ''}`}
                                        onClick={() => setModelType('seed')}
                                    >Seed (bootstrap)</button>
                                    <button
                                        type="button"
                                        className={`mtp-tab ${modelType === 'main' ? 'mtp-tab--active' : ''}`}
                                        onClick={() => setModelType('main')}
                                    >Main</button>
                                </div>

                                <div className="mtp-model-row">
                                    <label className="mtp-model-label">Starting Weights (YOLO-seg)</label>
                                    <select
                                        className="mtp-model-select"
                                        value={selectedModel}
                                        onChange={e => setSelectedModel(e.target.value)}
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
                                    >
                                        <option value={-1}>Auto (recommended)</option>
                                        {[2, 4, 8, 16, 32].map(b => <option key={b} value={b}>{b}</option>)}
                                    </select>
                                </div>
                                {imgsz > 640 && (
                                    <div className="mtp-warning" style={{ marginTop: 6 }}>
                                        ⚠️ Image size {imgsz} uses significantly more VRAM. With 20 GB GPU, keep batch ≤ 4 or use Auto batch. Recommended: 640 for safe training.
                                    </div>
                                )}

                                <div className="mtp-config-rows" style={{ marginTop: 12 }}>
                                    {[
                                        ['Classes', project.classes?.length > 0 ? project.classes.join(', ') : 'dynamic'],
                                        ['Training Images', stats?.annotated_images ?? '—'],
                                        ['Max Parallel Jobs', MAX_PARALLEL],
                                    ].map(([k, v]) => (
                                        <div key={k} className="mtp-config-row">
                                            <span className="mtp-config-key">{k}</span>
                                            <span className="mtp-config-val">{v}</span>
                                        </div>
                                    ))}
                                </div>

                                <div className="mtp-toggle-row" style={{ marginTop: 12 }}>
                                    <label className="mtp-toggle-label">
                                        <input
                                            type="checkbox"
                                            className="mtp-toggle-check"
                                            checked={preprocess}
                                            onChange={e => setPreprocess(e.target.checked)}
                                        />
                                        <span className="mtp-toggle-slider" />
                                        <span className="mtp-toggle-text">CLAHE contrast preprocessing</span>
                                    </label>
                                </div>
                                {!preprocess && (
                                    <div className="mtp-warning" style={{ marginTop: 8 }}>
                                        Preprocessing disabled — raw images will be used as-is. Detection of subtle brightness-based defects may be less accurate.
                                    </div>
                                )}

                                {/* ── Augmentation Settings ── */}
                                <div style={{ marginTop: 14 }}>
                                    <button onClick={() => setShowAugSettings(v => !v)} style={{ background: 'none', border: '1px solid #e5e5e5', borderRadius: 6, padding: '5px 10px', cursor: 'pointer', fontSize: 12, color: '#555', display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <span>{showAugSettings ? '▼' : '▶'}</span> Augmentation Settings
                                    </button>
                                    {showAugSettings && (
                                        <div style={{ marginTop: 10, padding: '12px 14px', background: '#f9f9f9', borderRadius: 8, border: '1px solid #e5e5e5' }}>

                                            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 }}>Color &amp; Lighting</p>

                                            <div style={{ marginBottom: 10 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Hue (hsv_h): <strong>{augHsvH}</strong></span>
                                                <input type="range" min={0.0} max={0.5} step={0.005} value={augHsvH} onChange={e => setAugHsvH(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0.0 (off)</span><span>0.015 (default)</span><span>0.5 (max)</span></div>
                                            </div>

                                            <div style={{ marginBottom: 10 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Saturation (hsv_s): <strong>{augHsvS}</strong></span>
                                                <input type="range" min={0.0} max={1.0} step={0.05} value={augHsvS} onChange={e => setAugHsvS(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0.0 (off)</span><span>0.3 (default)</span><span>1.0 (max)</span></div>
                                            </div>

                                            <div style={{ marginBottom: 14 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Brightness (hsv_v): <strong>{augHsvV}</strong></span>
                                                <input type="range" min={0.0} max={0.6} step={0.05} value={augHsvV} onChange={e => setAugHsvV(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0.0 (off)</span><span>0.4 (default)</span><span>0.6 (max)</span></div>
                                            </div>

                                            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 }}>Geometry</p>

                                            <div style={{ marginBottom: 10 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Rotation (degrees): <strong>{augDegrees}°</strong> <span title="Auto-capped to 3° server-side for single-character classes — a rotated dot-mask can flip a 6 into a 9." style={{ cursor: 'help', color: '#aaa', fontSize: 11 }}>ⓘ</span></span>
                                                <input type="range" min={0} max={45} step={1} value={augDegrees} onChange={e => setAugDegrees(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0° (off)</span><span>10° (default)</span><span>45° (max)</span></div>
                                            </div>

                                            <div style={{ marginBottom: 10 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Translation (translate): <strong>{augTranslate}</strong></span>
                                                <input type="range" min={0.0} max={0.3} step={0.05} value={augTranslate} onChange={e => setAugTranslate(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0.0 (off)</span><span>0.1 (default)</span><span>0.3 (max)</span></div>
                                            </div>

                                            <div style={{ marginBottom: 14 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Scale / Zoom (scale): <strong>{augScale}</strong></span>
                                                <input type="range" min={0.0} max={0.9} step={0.05} value={augScale} onChange={e => setAugScale(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0.0 (off)</span><span>0.4 (default)</span><span>0.9 (max)</span></div>
                                            </div>

                                            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 }}>Flip</p>

                                            <div className="mtp-toggle-row">
                                                <label className="mtp-toggle-label">
                                                    <input type="checkbox" className="mtp-toggle-check" checked={augFliplr} onChange={e => setAugFliplr(e.target.checked)} />
                                                    <span className="mtp-toggle-slider" />
                                                    <span className="mtp-toggle-text">Flip Left/Right <span title="Auto-forced off server-side for single-character classes — mirroring flips a character's identity (e.g. S)." style={{ cursor: 'help', color: '#aaa', fontSize: 11 }}>ⓘ</span></span>
                                                </label>
                                            </div>
                                            <div className="mtp-toggle-row" style={{ marginTop: 8, marginBottom: 14 }}>
                                                <label className="mtp-toggle-label">
                                                    <input type="checkbox" className="mtp-toggle-check" checked={augFlipud} onChange={e => setAugFlipud(e.target.checked)} />
                                                    <span className="mtp-toggle-slider" />
                                                    <span className="mtp-toggle-text">Flip Upside Down <span title="Auto-forced off server-side for single-character classes — e.g. M upside down looks like W." style={{ cursor: 'help', color: '#aaa', fontSize: 11 }}>ⓘ</span></span>
                                                </label>
                                            </div>

                                            <p style={{ fontSize: 11, fontWeight: 600, color: '#888', margin: '0 0 8px', textTransform: 'uppercase', letterSpacing: 0.5 }}>Mixing</p>

                                            <div className="mtp-toggle-row">
                                                <label className="mtp-toggle-label">
                                                    <input type="checkbox" className="mtp-toggle-check" checked={augMosaic} onChange={e => setAugMosaic(e.target.checked)} />
                                                    <span className="mtp-toggle-slider" />
                                                    <span className="mtp-toggle-text">Mosaic</span>
                                                </label>
                                            </div>

                                            <div style={{ marginTop: 10 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Mixup: <strong>{augMixup}</strong></span>
                                                <input type="range" min={0.0} max={0.3} step={0.05} value={augMixup} onChange={e => setAugMixup(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0.0 (off)</span><span>0.15</span><span>0.3 (max)</span></div>
                                            </div>

                                            <div style={{ marginTop: 10 }}>
                                                <span style={{ fontSize: 12, color: '#555' }}>Copy Paste: <strong>{augCopyPaste}</strong></span>
                                                <input type="range" min={0.0} max={0.2} step={0.01} value={augCopyPaste} onChange={e => setAugCopyPaste(parseFloat(e.target.value))} style={{ width: '100%', marginTop: 4 }} />
                                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}><span>0.0 (off)</span><span>0.05 (default)</span><span>0.2 (max)</span></div>
                                            </div>

                                        </div>
                                    )}
                                </div>
                            </section>

                            <section className="mtp-section">
                                <p className="mtp-section-title">Worker Setup Required</p>
                                <div className="mtp-worker-box">
                                    <p className="mtp-worker-desc">Open a <strong>new terminal</strong> in <code>backend/</code> and run:</p>
                                    <div className="mtp-cmd-block">
                                        <code>celery -A app.tasks.celery_app:celery_app worker --loglevel=info</code>
                                        <button className="mtp-cmd-copy" onClick={() => navigator.clipboard.writeText('celery -A app.tasks.celery_app:celery_app worker --loglevel=info')} title="Copy"><Copy size={14} /></button>
                                    </div>
                                </div>
                            </section>
                        </>
                    )}

                    {/* ═══════════ JOBS VIEW ═══════════ */}
                    {view === 'jobs' && (
                        jobs.length === 0 ? (
                            <div className="mtp-jobs-empty">
                                <span className="mtp-jobs-empty-icon"><Inbox size={32} /></span>
                                <p>No training jobs yet.</p>
                                <p className="mtp-jobs-empty-sub">Click <strong>Train Segmentation Model</strong> below to launch one.</p>
                            </div>
                        ) : (
                            <div className="mtp-jobs-layout">

                                <div className="mtp-jobs-list">
                                    {jobs.some(j => ['FAILURE','REVOKED','NO_WORKER'].includes(j.status)) && (
                                        <button
                                            onClick={clearFailedJobs}
                                            style={{ width: '100%', marginBottom: 6, padding: '4px 8px', fontSize: 11, background: 'rgba(220,20,60,0.07)', border: '1px solid rgba(220,20,60,0.2)', borderRadius: 6, color: '#dc143c', cursor: 'pointer' }}
                                        >Clear all failed</button>
                                    )}
                                    {[...jobs].reverse().map((job, idx) => {
                                        const info = STATUS_LABEL[job.status] || { label: job.status, cls: 'badge--pending' };
                                        const isDone = ['FAILURE','REVOKED','NO_WORKER','SUCCESS'].includes(job.status);
                                        return (
                                            <div
                                                key={job.id}
                                                className={`mtp-job-item ${activeJob?.id === job.id ? 'mtp-job-item--active' : ''}`}
                                                onClick={() => setActiveJobId(job.id)}
                                            >
                                                <div className="mtp-job-item-top">
                                                    <span className="mtp-job-num">#{jobs.length - idx}</span>
                                                    <span className={`mtp-job-badge ${info.cls}`}>{info.label}</span>
                                                    {isDone && (
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); removeJob(job); }}
                                                            title="Remove"
                                                            style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: '#999', padding: '0 2px', lineHeight: 1 }}
                                                        >✕</button>
                                                    )}
                                                </div>
                                                <div className="mtp-job-item-model">{job.modelName} ({job.modelType})</div>
                                                <div className="mtp-job-item-time">{fmtTime(job.startedAt)}</div>
                                                {job.status === 'STARTED' && job.epochMeta?.epoch != null && (
                                                    <div className="mtp-job-item-epoch">
                                                        {job.epochMeta.epoch}/{job.epochMeta.total_epochs}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>

                                {activeJob && (
                                    <div className="mtp-job-detail">
                                        <div className="mtp-job-detail-header">
                                            {(() => {
                                                const info = STATUS_LABEL[activeJob.status] || { label: activeJob.status, cls: 'badge--pending' };
                                                return <span className={`mtp-job-badge mtp-job-badge--lg ${info.cls}`}>{info.label}</span>;
                                            })()}
                                            <span className="mtp-job-detail-time">{fmtTime(activeJob.startedAt)}</span>
                                            {(activeJob.status === 'PENDING' || activeJob.status === 'STARTED') && (
                                                <span className="mtp-running-badge">● Live</span>
                                            )}
                                        </div>

                                        {activeJob.modelName && (
                                            <div className="mtp-job-model-tag">
                                                <span className="mtp-job-model-icon"><Scissors size={12} /></span>
                                                {activeJob.modelName} ({activeJob.modelType})
                                            </div>
                                        )}

                                        {activeJob.taskId && (
                                            <div className="mtp-job-taskid">
                                                <span className="mtp-job-taskid-label">Task ID</span>
                                                <span className="mtp-job-taskid-val">{activeJob.taskId}</span>
                                            </div>
                                        )}

                                        {activeJob.status === 'QUEUED' && (
                                            <div className="mtp-info" style={{ marginBottom: 12 }}>
                                                ⏳ Queued — will start when a slot opens (max {MAX_PARALLEL} parallel).
                                            </div>
                                        )}

                                        {activeJob.status === 'STARTED' && (
                                            <PreprocessingProgress meta={activeJob.epochMeta} />
                                        )}

                                        {(activeJob.status === 'STARTED' || activeJob.status === 'SUCCESS') && (
                                            <EpochProgress meta={activeJob.epochMeta} />
                                        )}

                                        {(() => {
                                            const split = activeJob.epochMeta?.split || activeJob.result?.split;
                                            if (!split) return null;
                                            const total = (split.train || 0) + (split.val || 0) + (split.test || 0);
                                            return (
                                                <div className="mtp-split-row">
                                                    <span className="mtp-split-label">Dataset split</span>
                                                    <div className="mtp-split-badges">
                                                        <span className="mtp-split-badge mtp-split-badge--train">Train&nbsp;{split.train}</span>
                                                        <span className="mtp-split-badge mtp-split-badge--val">Val&nbsp;{split.val}</span>
                                                        {split.test > 0 && (
                                                            <span className="mtp-split-badge mtp-split-badge--test">Test&nbsp;{split.test}</span>
                                                        )}
                                                        <span className="mtp-split-badge mtp-split-badge--total">Total&nbsp;{total}</span>
                                                    </div>
                                                </div>
                                            );
                                        })()}

                                        {activeJob.epochMeta?.history?.length > 0 && (
                                            <div className="charts-section">
                                                <LossChart history={activeJob.epochMeta.history} />
                                                <MapChart history={activeJob.epochMeta.history} />
                                            </div>
                                        )}

                                        <div className="mtp-section-header" style={{ marginBottom: 6 }}>
                                            <span className="mtp-section-title" style={{ margin: 0 }}>Log</span>
                                        </div>
                                        <div className="mtp-logs">
                                            {activeJob.logs.map((line, i) => (
                                                <div key={i} className="mtp-log-line">{line}</div>
                                            ))}
                                            <div ref={logsEndRef} />
                                        </div>

                                        {activeJob.status === 'SUCCESS' && activeJob.result && (
                                            <div className="mtp-result-card" style={{ marginTop: 12 }}>
                                                <span className="mtp-result-icon">🏆</span>
                                                <div>
                                                    <p className="mtp-result-title">Segmentation training complete</p>
                                                    <p className="mtp-result-path">{activeJob.result.model_path}</p>
                                                </div>
                                            </div>
                                        )}

                                        {activeJob.status === 'SUCCESS' && activeJob.result?.metrics && Object.keys(activeJob.result.metrics).length > 0 && (
                                            <div style={{ marginTop: 12 }}>
                                                <p className="mtp-section-title" style={{ marginBottom: 8 }}>Final Metrics</p>
                                                <div className="mtp-config-rows">
                                                    {Object.entries(activeJob.result.metrics)
                                                        .filter(([k]) => k !== 'epoch')
                                                        .map(([k, v]) => (
                                                            <div key={k} className="mtp-config-row">
                                                                <span className="mtp-config-key">{k}</span>
                                                                <span className="mtp-config-val">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                                                            </div>
                                                        ))}
                                                </div>
                                            </div>
                                        )}

                                        {activeJob.status === 'NO_WORKER' && (
                                            <div className="mtp-worker-box mtp-worker-box--error" style={{ marginTop: 12 }}>
                                                <p className="mtp-worker-error">⚠️ Worker not detected. Start it, then try again.</p>
                                                <div className="mtp-cmd-block">
                                                    <code>celery -A app.tasks.celery_app:celery_app worker --loglevel=info</code>
                                                    <button className="mtp-cmd-copy" onClick={() => navigator.clipboard.writeText('celery -A app.tasks.celery_app:celery_app worker --loglevel=info')}><Copy size={14} /></button>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )
                    )}
                </div>

                {/* ── Footer ── */}
                <div className="mtp-footer">
                    <button className="mtp-train-btn" onClick={handleTrain} disabled={!readyToTrain || launching}>
                        {btnLabel()}
                    </button>
                    {anyRunning && (
                        <button className="mtp-stop-btn" onClick={handleForceStopAll} title="Stop all running and queued jobs, purge queue">
                            <Square size={14} /> Force Stop All
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SegTrainingPanel;
