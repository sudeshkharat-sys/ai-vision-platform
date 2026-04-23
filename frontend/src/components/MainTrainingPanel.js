import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Brain, X, RefreshCw, Download, Inbox, Square, ChevronLeft, Copy } from 'lucide-react';
import { YOLO_MODEL_GROUPS, DEFAULT_MAIN_MODEL } from '../constants/yoloModels';
import './MainTrainingPanel.css';
import logoImg from '../logo.png';

import { API_URL } from '../config';
const POLL_INTERVAL = 3000;
const MAX_PARALLEL  = 2;
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

// ── Helpers ────────────────────────────────────────────────────────
const makeJob = (taskId, modelName) => ({
    id: Date.now(),
    taskId,
    modelName,
    status: 'PENDING',
    logs: [`[ID] Task ID: ${taskId}`, '[...] Waiting for worker...'],
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

// ── Chart tooltip ──────────────────────────────────────────────────
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

// ── Loss chart ─────────────────────────────────────────────────────
const LossChart = ({ history }) => {
    if (!history?.length) return null;
    const hasBox = history.some(h => h.box_loss != null);
    const hasCls = history.some(h => h.cls_loss != null);
    const hasDfl = history.some(h => h.dfl_loss != null);
    if (!hasBox && !hasCls && !hasDfl) return null;
    return (
        <div className="chart-wrap">
            <p className="chart-title">Training Loss</p>
            <ResponsiveContainer width="100%" height={160}>
                <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis dataKey="epoch" tick={{ fill: '#666666', fontSize: 10 }}
                        label={{ value: 'Epoch', position: 'insideBottom', fill: '#666666', fontSize: 10, offset: -1 }} />
                    <YAxis tick={{ fill: '#666666', fontSize: 10 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 10, color: '#666666' }} />
                    {hasBox && <Line type="monotone" dataKey="box_loss" name="Box" stroke="#dc143c" dot={false} strokeWidth={1.5} />}
                    {hasCls && <Line type="monotone" dataKey="cls_loss" name="Cls" stroke="#f59e0b" dot={false} strokeWidth={1.5} />}
                    {hasDfl && <Line type="monotone" dataKey="dfl_loss" name="DFL" stroke="#ec4899" dot={false} strokeWidth={1.5} />}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

// ── mAP chart ──────────────────────────────────────────────────────
const MapChart = ({ history }) => {
    if (!history?.length) return null;
    const hasMap50 = history.some(h => h['mAP50'] != null || h['mAP50(B)'] != null);
    const hasMap95 = history.some(h => h['mAP50-95'] != null || h['mAP50-95(B)'] != null);
    if (!hasMap50 && !hasMap95) return null;
    const data = history.map(h => ({
        epoch: h.epoch,
        mAP50: h['mAP50'] ?? h['mAP50(B)'] ?? null,
        'mAP50-95': h['mAP50-95'] ?? h['mAP50-95(B)'] ?? null,
    }));
    return (
        <div className="chart-wrap">
            <p className="chart-title">Validation mAP</p>
            <ResponsiveContainer width="100%" height={160}>
                <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                    <XAxis dataKey="epoch" tick={{ fill: '#666666', fontSize: 10 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: '#666666', fontSize: 10 }} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 10, color: '#666666' }} />
                    {hasMap50 && <Line type="monotone" dataKey="mAP50" name="mAP50" stroke="#4ade80" dot={false} strokeWidth={1.5} />}
                    {hasMap95 && <Line type="monotone" dataKey="mAP50-95" name="mAP50-95" stroke="#38bdf8" dot={false} strokeWidth={1.5} />}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

// ── PreprocessingProgress ──────────────────────────────────────────
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

// ── Epoch progress bar ─────────────────────────────────────────────
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
                <div className="epoch-bar-fill mtp-bar-fill" style={{ width: `${pct}%` }} />
            </div>
        </div>
    );
};

// ══════════════════════════════════════════════════════════════════
//  MainTrainingPanel
// ══════════════════════════════════════════════════════════════════
const MainTrainingPanel = ({ project, onClose }) => {
    const [stats, setStats]               = useState(null);
    const [statsLoading, setStatsLoading] = useState(true);
    const [modelStatus, setModelStatus]   = useState(null);
    const [jobs, setJobs]                 = useState([]);
    const [activeJobId, setActiveJobId]   = useState(null);
    const [launching, setLaunching]       = useState(false);
    const [view, setView]                 = useState('detail');

    // Config state
    const [selectedModel, setSelectedModel]     = useState(DEFAULT_MAIN_MODEL);
    const [epochs, setEpochs]                   = useState(60);
    const [useSeedWeights, setUseSeedWeights]   = useState(true);
    const [imgsz, setImgsz]                     = useState(640);
    const [batch, setBatch]                     = useState(-1);
    const [preprocess, setPreprocess]           = useState(true);
    const [clahePreview, setClahePreview]       = useState(null);
    const [previewLoading, setPreviewLoading]   = useState(false);

    const logsEndRef = useRef(null);
    const pollRef    = useRef({});
    const jobsRef    = useRef(jobs);
    const queueRef   = useRef([]);

    useEffect(() => { jobsRef.current = jobs; }, [jobs]);

    // ── Load stats + model status ──────────────────────────────────
    const loadData = useCallback(() => {
        setStatsLoading(true);
        Promise.all([
            axios.get(`${API_URL}/pipeline/training-stats/${project.id}`),
            axios.get(`${API_URL}/pipeline/model-status/${project.id}`),
        ])
            .then(([statsRes, modelRes]) => {
                setStats(statsRes.data);
                setModelStatus(modelRes.data);
            })
            .catch(() => {})
            .finally(() => setStatsLoading(false));
    }, [project.id]);

    // ── CLAHE preview ──────────────────────────────────────────────
    const loadClahePreview = useCallback(() => {
        setPreviewLoading(true);
        axios.get(`${API_URL}/pipeline/clahe-preview/${project.id}`)
            .then(res => setClahePreview(res.data))
            .catch(() => setClahePreview(null))
            .finally(() => setPreviewLoading(false));
    }, [project.id]);

    // ── Load persisted jobs from DB ────────────────────────────────
    const loadPersistedJobs = useCallback((resumePollingFn) => {
        axios.get(`${API_URL}/pipeline/jobs/${project.id}?job_type=main_training`)
            .then(async res => {
                if (!res.data.length) return;
                const DB_STATUS = { pending: 'PENDING', started: 'STARTED', success: 'SUCCESS', failure: 'FAILURE' };
                const loaded = res.data.map(j => ({
                    id:        j.id,
                    taskId:    j.id,
                    modelName: j.result_meta?.modelName || DEFAULT_MAIN_MODEL,
                    status:    DB_STATUS[j.status] || 'PENDING',
                    logs:      j.result_meta?.logs      || [`📋  Task ID: ${j.id}`],
                    epochMeta: j.result_meta?.epochMeta || null,
                    result:    j.result_meta?.result    || null,
                    error:     j.result_meta?.error     || null,
                    startedAt: new Date(j.created_at),
                }));

                // ── Reconcile ALL active jobs in parallel BEFORE rendering ──
                // Single setJobs call after all checks complete — no flicker.
                const reconciled = await Promise.all(loaded.map(async (job) => {
                    if (job.status !== 'PENDING' && job.status !== 'STARTED') return job;

                    try {
                        const sr = await axios.get(`${API_URL}/pipeline/task-status/${job.taskId}`);
                        const { status: celery, result, error } = sr.data;

                        if (celery === 'SUCCESS') {
                            const newLogs = [...job.logs, '✅  Main training complete! (completed while panel was closed)'];
                            const finalMeta = result?.history?.length
                                ? { ...job.epochMeta, history: result.history, epoch: result.history.length }
                                : job.epochMeta;
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'success',
                                result_meta: { logs: newLogs, epochMeta: finalMeta, result, modelName: job.modelName },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            loadData();
                            return { ...job, status: 'SUCCESS', result, logs: newLogs, epochMeta: finalMeta };

                        } else if (celery === 'FAILURE') {
                            const newLogs = [...job.logs, `❌  Failed: ${error || 'Unknown error'}`];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { logs: newLogs, error, modelName: job.modelName },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...job, status: 'FAILURE', error, logs: newLogs };

                        } else if (celery === 'PENDING' && job.status === 'STARTED') {
                            // DB says "started" but Celery shows PENDING — result expired.
                            const newLogs = [...job.logs,
                                '⚠️  Task result expired (panel was closed before it finished).',
                                '    The main training run may have completed — check for a saved model.',
                            ];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { logs: newLogs, modelName: job.modelName },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...job, status: 'FAILURE', logs: newLogs };

                        } else {
                            // Genuinely still running — start the normal poll loop
                            resumePollingFn(job.taskId);
                            return job;
                        }
                    } catch {
                        // Can't reach status endpoint — resume polling so it retries
                        resumePollingFn(job.taskId);
                        return job;
                    }
                }));

                // Single state update — no flicker
                setJobs(reconciled);
                setActiveJobId(reconciled[reconciled.length - 1].id);
            })
            .catch(() => {});
    }, [project.id, loadData]);

    useEffect(() => {
        loadData();
        loadPersistedJobs(startPolling);
        const polls = pollRef.current;
        return () => { Object.values(polls).forEach(clearInterval); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [loadData, loadPersistedJobs]);

    // auto-scroll log
    useEffect(() => {
        if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [jobs, activeJobId]);

    const runningCount = () =>
        jobsRef.current.filter(j => j.status === 'PENDING' || j.status === 'STARTED').length;

    // ── Polling ───────────────────────────────────────────────────
    const startPolling = useCallback((taskId) => {
        if (pollRef.current[taskId]) clearInterval(pollRef.current[taskId]);
        let pendingTicks    = 0;
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
                            else if (last !== '⚙️  Main training in progress…')
                                newLogs = [...newLogs, '⚙️  Main training in progress…'];
                        }
                        return { ...j, status: 'STARTED', logs: newLogs, epochMeta: meta || j.epochMeta };
                    }

                    if (status === 'SUCCESS') {
                        clearInterval(pollRef.current[taskId]);
                        delete pollRef.current[taskId];
                        newLogs = [...newLogs, '✅  Main training complete!',
                            result?.model_path ? `📦  Model: ${result.model_path}` : ''].filter(Boolean);
                        loadData();
                        setTimeout(() => dispatchQueued(), 100);
                        const finalMeta = result?.history?.length
                            ? { ...j.epochMeta, history: result.history, epoch: result.history.length }
                            : j.epochMeta;
                        axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                            status: 'success',
                            result_meta: {
                                logs: newLogs, epochMeta: finalMeta, result,
                                modelName: j.modelName,
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
                                modelName: j.modelName,
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
    }, [loadData]);

    // ── Dispatch queued job ────────────────────────────────────────
    const dispatchQueued = useCallback(async () => {
        if (runningCount() >= MAX_PARALLEL) return;
        const next = queueRef.current.shift();
        if (!next) return;
        try {
            const res = await axios.post(`${API_URL}/pipeline/train-main/${next.projectId}`, {
                model_name: next.modelName, epochs: next.epochs,
                use_seed_weights: next.useSeedWeights, imgsz: next.imgsz,
                preprocess: next.preprocess, batch: next.batch,
            });
            const taskId = res.data.task_id;
            const logs = [`📋  Task ID: ${taskId}`, '⏳  Waiting for worker…'];
            setJobs(prev => prev.map(j =>
                j.id === next.jobId
                    ? { ...j, taskId, status: 'PENDING', logs }
                    : j
            ));
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: taskId, project_id: next.projectId, job_type: 'main_training',
                result_meta: { logs, startedAt: new Date().toISOString(), modelName: next.modelName },
            }).catch(() => {});
            startPolling(taskId);
        } catch {
            setJobs(prev => prev.map(j =>
                j.id === next.jobId
                    ? { ...j, status: 'FAILURE', logs: [...j.logs, '[ERR] Failed to queue job.'] }
                    : j
            ));
        }
    }, [startPolling]);

    // ── Stop running job ───────────────────────────────────────────
    const handleStop = async () => {
        const job = activeJob || jobs.find(j => j.status === 'PENDING' || j.status === 'STARTED');
        if (!job?.taskId) return;
        try {
            await axios.post(`${API_URL}/pipeline/cancel/${job.taskId}`);
        } catch { /* ignore network errors */ }
        if (pollRef.current[job.taskId]) {
            clearInterval(pollRef.current[job.taskId]);
            delete pollRef.current[job.taskId];
        }
        setJobs(prev => prev.map(j =>
            j.taskId === job.taskId
                ? { ...j, status: 'REVOKED', logs: [...j.logs, '🛑  Stopped by user.'] }
                : j
        ));
    };

    // ── Launch ─────────────────────────────────────────────────────
    const handleTrain = async () => {
        setLaunching(true);
        setView('jobs');

        if (runningCount() >= MAX_PARALLEL) {
            const placeholder = {
                id: Date.now(), taskId: null, status: 'QUEUED',
                modelName: selectedModel,
                logs: ['[INFO] Job queued - waiting for a free slot...'],
                epochMeta: null, result: null, error: null, startedAt: new Date(),
            };
            queueRef.current.push({
                jobId: placeholder.id, projectId: project.id,
                modelName: selectedModel, epochs, useSeedWeights, imgsz, preprocess, batch,
            });
            setJobs(prev => [...prev, placeholder]);
            setActiveJobId(placeholder.id);
            setLaunching(false);
            return;
        }

        try {
            const res = await axios.post(`${API_URL}/pipeline/train-main/${project.id}`, {
                model_name: selectedModel, epochs, use_seed_weights: useSeedWeights, imgsz, preprocess, batch,
            });
            const taskId = res.data.task_id;
            const job = makeJob(taskId, selectedModel);
            setJobs(prev => [...prev, job]);
            setActiveJobId(job.id);
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: taskId, project_id: project.id, job_type: 'main_training',
                result_meta: { logs: job.logs, startedAt: job.startedAt.toISOString(), modelName: selectedModel },
            }).catch(() => {});
            startPolling(taskId);
        } catch {
            const failJob = {
                id: Date.now(), taskId: null, status: 'FAILURE',
                modelName: selectedModel,
                logs: ['[ERR] Failed to queue main training job.'],
                epochMeta: null, result: null, error: 'Failed to queue', startedAt: new Date(),
            };
            setJobs(prev => [...prev, failJob]);
            setActiveJobId(failJob.id);
        } finally {
            setLaunching(false);
        }
    };

    // ── Derived ────────────────────────────────────────────────────
    const activeJob    = jobs.find(j => j.id === activeJobId) || jobs[jobs.length - 1] || null;
    const anyRunning   = jobs.some(j => j.status === 'PENDING' || j.status === 'STARTED');
    const hasSeed      = modelStatus?.has_seed_model;
    const hasMain      = modelStatus?.has_main_model;
    const readyToTrain = stats?.ready_to_train && (!useSeedWeights || hasSeed);

    const btnLabel = () => {
        if (launching) return 'Queuing…';
        if (runningCount() >= MAX_PARALLEL) return <><Download size={16} /> Queue Job</>;
        return <><Brain size={16} /> Start Main Training</>;
    };

    return (
        <div className="mtp-overlay" onClick={onClose}>
            <div className="mtp-panel" onClick={e => e.stopPropagation()}>

                {/* ── Header ── */}
                <div className="mtp-header">
                    <div className="mtp-header-left">
                        <span className="mtp-header-icon"><Brain size={20} /></span>
                        <div>
                            <h2 className="mtp-title">Train Main Model</h2>
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

                    {/* ═════════════ DETAIL VIEW ═════════════ */}
                    {view === 'detail' && (
                        <>
                            {/* Dataset overview */}
                            <section className="mtp-section">
                                <div className="mtp-section-header">
                                    <span className="mtp-section-title">Dataset Overview</span>
                                    <button className="mtp-refresh" onClick={loadData} title="Refresh"><RefreshCw size={15} /></button>
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
                                            <div className="mtp-progress-bar">
                                                <div className="mtp-progress-fill" style={{ width: stats.total_images > 0 ? `${(stats.annotated_images / stats.total_images) * 100}%` : '0%' }} />
                                            </div>
                                        </div>

                                        {/* Model availability status */}
                                        <div className="mtp-model-status-row">
                                            <div className={`mtp-model-chip ${hasSeed ? 'mtp-model-chip--ok' : 'mtp-model-chip--miss'}`}>
                                                {hasSeed ? '✅' : '⚠️'} Seed Model
                                            </div>
                                            <div className={`mtp-model-chip ${hasMain ? 'mtp-model-chip--ok' : 'mtp-model-chip--none'}`}>
                                                {hasMain ? '✅' : '○'} Main Model
                                            </div>
                                        </div>

                                        {!stats.ready_to_train && (
                                            <div className="mtp-warning">⚠️ Annotate at least 1 image before training.</div>
                                        )}
                                        {useSeedWeights && !hasSeed && (
                                            <div className="mtp-warning">⚠️ Seed model required when "Fine-tune from seed" is on. Train seed model first, or disable fine-tuning.</div>
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
                                    </>
                                ) : <p className="mtp-error-text">Failed to load stats.</p>}
                            </section>

                            {/* Training config */}
                            <section className="mtp-section">
                                <p className="mtp-section-title">Training Config</p>

                                {/* YOLO architecture */}
                                <div className="mtp-model-row">
                                    <label className="mtp-model-label">YOLO Architecture</label>
                                    <select
                                        className="mtp-model-select"
                                        value={selectedModel}
                                        onChange={e => setSelectedModel(e.target.value)}
                                    >
                                        {YOLO_MODEL_GROUPS.map(group => (
                                            <optgroup key={group.family} label={`${group.family}${group.note ? ` (${group.note})` : ''}`}>
                                                {group.models.map(m => (
                                                    <option key={m.value} value={m.value}>
                                                        {m.label}  [{m.params}]
                                                    </option>
                                                ))}
                                            </optgroup>
                                        ))}
                                    </select>
                                </div>

                                {/* Fine-tune from seed toggle */}
                                <div className="mtp-toggle-row">
                                    <label className="mtp-toggle-label">
                                        <input
                                            type="checkbox"
                                            className="mtp-toggle-check"
                                            checked={useSeedWeights}
                                            onChange={e => setUseSeedWeights(e.target.checked)}
                                        />
                                        <span className="mtp-toggle-slider" />
                                        <span className="mtp-toggle-text">
                                            Fine-tune from seed model
                                            <span className="mtp-toggle-hint"> (faster convergence, recommended)</span>
                                        </span>
                                    </label>
                                </div>

                                {/* Epochs */}
                                <div className="mtp-model-row">
                                    <label className="mtp-model-label">
                                        Epochs
                                        <span className="mtp-model-hint">More = better accuracy, longer training</span>
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

                                {/* Image size */}
                                <div className="mtp-model-row">
                                    <label className="mtp-model-label">
                                        Image Size
                                        <span className="mtp-model-hint"> (larger = slower, needs more VRAM)</span>
                                    </label>
                                    <select
                                        className="mtp-model-select mtp-model-select--sm"
                                        value={imgsz}
                                        onChange={e => setImgsz(Number(e.target.value))}
                                    >
                                        {[320, 416, 512, 640, 768, 1024, 1280].map(s => (
                                            <option key={s} value={s}>{s} × {s}{s === 640 ? ' (recommended)' : ''}</option>
                                        ))}
                                    </select>
                                </div>

                                {/* Batch size */}
                                <div className="mtp-model-row">
                                    <label className="mtp-model-label">
                                        Batch Size
                                        <span className="mtp-model-hint"> (Auto finds max that fits in VRAM)</span>
                                    </label>
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

                                {/* Static info */}
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

                                {/* ── CLAHE preprocessing toggle ── */}
                                <div className="mtp-toggle-row" style={{ marginTop: 12 }}>
                                    <label className="mtp-toggle-label">
                                        <input
                                            type="checkbox"
                                            className="mtp-toggle-check"
                                            checked={preprocess}
                                            onChange={e => {
                                                setPreprocess(e.target.checked);
                                                if (e.target.checked && !clahePreview) loadClahePreview();
                                            }}
                                        />
                                        <span className="mtp-toggle-slider" />
                                        <span className="mtp-toggle-text">
                                            CLAHE contrast preprocessing
                                            <span className="mtp-toggle-hint"> (recommended for bright-feature inspection)</span>
                                        </span>
                                    </label>
                                </div>

                                {/* ── Before / After preview ── */}
                                {preprocess && (
                                    <div style={{ marginTop: 10 }}>
                                        {clahePreview ? (
                                            <>
                                                <p style={{ fontSize: 11, color: '#666666', marginBottom: 6 }}>
                                                    Preview — <em>{clahePreview.filename}</em>
                                                </p>
                                                <div style={{ display: 'flex', gap: 8 }}>
                                                    <div style={{ flex: 1, textAlign: 'center' }}>
                                                        <p style={{ fontSize: 10, color: '#666666', marginBottom: 4 }}>Original</p>
                                                        <img src={clahePreview.original} alt="Original" style={{ width: '100%', borderRadius: 4, border: '1px solid #e5e5e5' }} />
                                                    </div>
                                                    <div style={{ flex: 1, textAlign: 'center' }}>
                                                        <p style={{ fontSize: 10, color: '#16a34a', marginBottom: 4 }}>After CLAHE</p>
                                                        <img src={clahePreview.enhanced} alt="CLAHE enhanced" style={{ width: '100%', borderRadius: 4, border: '1px solid #16a34a' }} />
                                                    </div>
                                                </div>
                                            </>
                                        ) : previewLoading ? (
                                            <div className="mtp-info" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                <div className="mtp-spinner" style={{ width: 12, height: 12 }} />
                                                <span>Loading preview from dataset…</span>
                                            </div>
                                        ) : (
                                            <div className="mtp-info" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                <span>⚡</span>
                                                <span>
                                                    <strong>CLAHE active</strong> — contrast enhancement will be applied to all training images.
                                                    {stats?.annotated_images > 0 && (
                                                        <button
                                                            onClick={loadClahePreview}
                                                            style={{ marginLeft: 8, background: 'none', border: 'none', color: '#dc143c', cursor: 'pointer', fontSize: 11, textDecoration: 'underline', padding: 0 }}
                                                        >
                                                            Show preview
                                                        </button>
                                                    )}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                )}
                                {!preprocess && (
                                    <div className="mtp-warning" style={{ marginTop: 8 }}>
                                        Preprocessing disabled — raw images will be used as-is. Detection of subtle brightness-based defects may be less accurate.
                                    </div>
                                )}
                            </section>

                            {/* Worker setup */}
                            <section className="mtp-section">
                                <p className="mtp-section-title">Worker Setup Required</p>
                                <div className="mtp-worker-box">
                                    <p className="mtp-worker-desc">Open a <strong>new terminal</strong> in <code>backend/</code> and run:</p>
                                    <div className="mtp-cmd-block">
                                        <code>celery -A app.tasks.celery_app:celery_app worker --loglevel=info</code>
                                        <button className="mtp-cmd-copy"
                                            onClick={() => navigator.clipboard.writeText('celery -A app.tasks.celery_app:celery_app worker --loglevel=info')}
                                            title="Copy"><Copy size={14} /></button>
                                    </div>
                                </div>
                            </section>
                        </>
                    )}

                    {/* ═════════════ JOBS VIEW ═════════════ */}
                    {view === 'jobs' && (
                        jobs.length === 0 ? (
                            <div className="mtp-jobs-empty">
                                <span className="mtp-jobs-empty-icon"><Inbox size={32} /></span>
                                <p>No main training jobs yet.</p>
                                <p className="mtp-jobs-empty-sub">Click <strong>Start Main Training</strong> below.</p>
                            </div>
                        ) : (
                            <div className="mtp-jobs-layout">

                                {/* Job list */}
                                <div className="mtp-jobs-list">
                                    {[...jobs].reverse().map((job, idx) => {
                                        const info = STATUS_LABEL[job.status] || { label: job.status, cls: 'badge--pending' };
                                        return (
                                            <div
                                                key={job.id}
                                                className={`mtp-job-item ${activeJob?.id === job.id ? 'mtp-job-item--active' : ''}`}
                                                onClick={() => setActiveJobId(job.id)}
                                            >
                                                <div className="mtp-job-item-top">
                                                    <span className="mtp-job-num">#{jobs.length - idx}</span>
                                                    <span className={`mtp-job-badge ${info.cls}`}>{info.label}</span>
                                                </div>
                                                <div className="mtp-job-item-model">{job.modelName}</div>
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

                                {/* Job detail */}
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

                                        {/* Model badge */}
                                        {activeJob.modelName && (
                                            <div className="mtp-job-model-tag">
                                                <span className="mtp-job-model-icon"><Brain size={12} /></span>
                                                {activeJob.modelName}
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

                                        {/* ── Preprocessing Progress ── */}
                                        {activeJob.status === 'STARTED' && (
                                            <PreprocessingProgress meta={activeJob.epochMeta} />
                                        )}

                                        {(activeJob.status === 'STARTED' || activeJob.status === 'SUCCESS') && (
                                            <EpochProgress meta={activeJob.epochMeta} />
                                        )}

                                        {/* ── Dataset Split ── */}
                                        {(() => {
                                            const split = activeJob.epochMeta?.split || activeJob.result?.split;
                                            if (!split) return null;
                                            const total = (split.train || 0) + (split.val || 0) + (split.test || 0);
                                            return (
                                                <div className="mtp-split-row">
                                                    <span className="mtp-split-label">Dataset split</span>
                                                    <div className="mtp-split-badges">
                                                        <span className="mtp-split-badge mtp-split-badge--train">
                                                            Train&nbsp;{split.train}
                                                        </span>
                                                        <span className="mtp-split-badge mtp-split-badge--val">
                                                            Val&nbsp;{split.val}
                                                        </span>
                                                        {split.test > 0 && (
                                                            <span className="mtp-split-badge mtp-split-badge--test">
                                                                Test&nbsp;{split.test}
                                                            </span>
                                                        )}
                                                        <span className="mtp-split-badge mtp-split-badge--total">
                                                            Total&nbsp;{total}
                                                        </span>
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
                                                    <p className="mtp-result-title">Main training complete</p>
                                                    <p className="mtp-result-path">{activeJob.result.model_path}</p>
                                                </div>
                                            </div>
                                        )}

                                        {activeJob.status === 'SUCCESS' && activeJob.result?.metrics &&
                                            Object.keys(activeJob.result.metrics).length > 0 && (
                                            <div style={{ marginTop: 12 }}>
                                                <p className="mtp-section-title" style={{ marginBottom: 8 }}>Final Metrics</p>
                                                <div className="mtp-config-rows">
                                                    {Object.entries(activeJob.result.metrics)
                                                        .filter(([k]) => k !== 'epoch')
                                                        .map(([k, v]) => (
                                                            <div key={k} className="mtp-config-row">
                                                                <span className="mtp-config-key">{k}</span>
                                                                <span className="mtp-config-val">
                                                                    {typeof v === 'number' ? v.toFixed(4) : String(v)}
                                                                </span>
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
                    <button
                        className="mtp-train-btn"
                        onClick={handleTrain}
                        disabled={!readyToTrain || launching}
                    >
                        {btnLabel()}
                    </button>
                    {anyRunning && (
                        <button className="mtp-stop-btn" onClick={handleStop} title="Stop training">
                            <Square size={14} /> Stop
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default MainTrainingPanel;
