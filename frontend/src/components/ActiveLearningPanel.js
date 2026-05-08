import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './ActiveLearningPanel.css';
import {
    Brain, Target, Layers, BarChart2, X, RefreshCw,
    Clipboard, Square, ChevronLeft, AlertTriangle,
    Check, Zap, Inbox, Play, Circle, Pencil,
} from 'lucide-react';

import { API_URL } from '../config';

// ── ETA formatter ────────────────────────────────────────────────
const fmtEta = (secs) => {
    if (secs == null || secs <= 0) return null;
    if (secs < 60)  return `~${secs}s left`;
    if (secs < 3600) return `~${Math.ceil(secs / 60)}m left`;
    return `~${Math.ceil(secs / 3600)}h left`;
};

const POLL_INTERVAL   = 2500;
const NO_WORKER_TICKS = 15; // ~37s

const STATUS_LABEL = {
    QUEUED:    { label: 'Queued',    cls: 'alp-badge--queued'  },
    PENDING:   { label: 'Pending',   cls: 'alp-badge--pending' },
    STARTED:   { label: 'Running',   cls: 'alp-badge--running' },
    SUCCESS:   { label: 'Done',      cls: 'alp-badge--done'    },
    FAILURE:   { label: 'Failed',    cls: 'alp-badge--fail'    },
    NO_WORKER: { label: 'No Worker', cls: 'alp-badge--fail'    },
};

const MODES = [
    {
        id: 'suggest',
        Icon: Target,
        label: 'Suggest to Annotate',
        desc: 'Find the most uncertain images to label manually',
    },
    {
        id: 'curriculum',
        Icon: Layers,
        label: 'Curriculum Auto-Annotate',
        desc: 'Smart auto-annotation with 3 confidence tiers',
    },
    {
        id: 'score',
        Icon: BarChart2,
        label: 'Score All Images',
        desc: 'Rank all pending images by uncertainty',
    },
];

const STRATEGIES = [
    { id: 'combined',   label: 'Combined',        desc: 'Confidence + Entropy + TTA (recommended)' },
    { id: 'confidence', label: 'Confidence',       desc: 'Lowest detection confidence first' },
    { id: 'entropy',    label: 'Entropy',          desc: 'Highest class confusion first' },
    { id: 'tta',        label: 'TTA Disagreement', desc: 'Most unstable predictions first' },
];

const fmtTime = (d) => new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const fmtPct  = (v) => `${Math.round(v * 100)}%`;

const makeJob = (taskId, mode, config) => ({
    id:        Date.now(),
    taskId,
    mode,
    config,
    status:    'PENDING',
    logs:      [`[ID] Task ID: ${taskId}`, `[MODE] ${MODES.find(m => m.id === mode)?.label}`],
    progress:  null,
    result:    null,
    error:     null,
    startedAt: new Date(),
});

// ── ProgressBar ──────────────────────────────────────────────────
const ProgressBar = ({ progress }) => {
    if (!progress?.total) return null;
    const pct     = Math.round((progress.current / progress.total) * 100);
    const eta     = fmtEta(progress.eta_secs);
    const skipped = progress.skipped_path ?? 0;
    return (
        <div className="alp-progress">
            <div className="alp-progress-header">
                <span className="alp-progress-label">
                    Image <strong>{progress.current}</strong> / {progress.total}
                </span>
                {eta && <span className="alp-progress-eta">{eta}</span>}
                <span className="alp-progress-pct">{pct}%</span>
            </div>
            <div className="alp-progress-bar">
                <div className="alp-progress-fill" style={{ width: `${pct}%` }} />
            </div>
            {/* Live curriculum stats */}
            {(progress.auto_accepted != null) && (
                <div className="alp-progress-stats">
                    <span className="alp-pstat alp-pstat--auto">✓ {progress.auto_accepted} auto</span>
                    <span className="alp-pstat alp-pstat--review">~ {progress.sent_to_review} review</span>
                    <span className="alp-pstat alp-pstat--skip">— {(progress.skipped_low_conf ?? 0) + (progress.skipped_no_det ?? 0)} skipped</span>
                    {skipped > 0 && <span className="alp-pstat alp-pstat--err">⚠ {skipped} not found</span>}
                </div>
            )}
        </div>
    );
};

// ── ScoreResultTable ─────────────────────────────────────────────
const ScoreResultTable = ({ items, label, onAnnotate }) => {
    if (!items || items.length === 0) return null;
    return (
        <div className="alp-result-table-wrap">
            <div className="alp-result-table-header">
                <div className="alp-result-table-title">
                    {label} <span className="alp-result-count">({items.length})</span>
                </div>
                {onAnnotate && (
                    <button className="alp-annotate-btn" onClick={() => onAnnotate(items.map(i => i.image_id))}>
                        <Pencil size={12} /> Annotate These
                    </button>
                )}
            </div>
            <div className="alp-result-table">
                {items.map((item, i) => (
                    <div key={item.image_id || i} className="alp-result-row">
                        <span className="alp-result-rank">#{item.rank || i + 1}</span>
                        <div className="alp-result-info">
                            <span className="alp-result-filename">{item.filename}</span>
                            {item.reason && <span className="alp-result-reason">{item.reason}</span>}
                        </div>
                        <span className="alp-result-score">
                            {typeof item.combined_score !== 'undefined'
                                ? item.combined_score.toFixed(3)
                                : typeof item.uncertainty !== 'undefined'
                                    ? item.uncertainty.toFixed(3)
                                    : '—'}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ── CurriculumResult ─────────────────────────────────────────────
const CurriculumResult = ({ result }) => {
    if (!result) return null;
    const skipped = (result.skipped_low_conf ?? 0) + (result.skipped_no_det ?? 0);
    return (
        <div className="alp-curriculum-result">
            <div className="alp-curriculum-result-title">Results</div>
            <div className="alp-curr-stats">
                <div className="alp-curr-stat alp-curr-stat--auto">
                    <span className="alp-curr-num">{result.auto_accepted ?? 0}</span>
                    <span className="alp-curr-label">Auto-Accepted</span>
                    <span className="alp-curr-sub">source: auto</span>
                </div>
                <div className="alp-curr-stat alp-curr-stat--review">
                    <span className="alp-curr-num">{result.sent_to_review ?? 0}</span>
                    <span className="alp-curr-label">Sent to Review</span>
                    <span className="alp-curr-sub">source: auto_review</span>
                </div>
                <div className="alp-curr-stat alp-curr-stat--skip">
                    <span className="alp-curr-num">{skipped}</span>
                    <span className="alp-curr-label">Skipped</span>
                    <span className="alp-curr-sub">manual needed</span>
                </div>
            </div>
            {skipped > 0 && (
                <p className="alp-curr-hint">
                    Skipped images are still pending — use <strong>Suggest to Annotate</strong> to find which ones to label manually.
                </p>
            )}
        </div>
    );
};

// ════════════════════════════════════════════════════════════
//  ActiveLearningPanel
// ════════════════════════════════════════════════════════════
const ActiveLearningPanel = ({ project, onClose, onAnnotationsUpdated, onAnnotateImages }) => {
    const [view, setView]               = useState('setup');
    const [modelStatus, setModelStatus] = useState(null);
    const [mode, setMode]               = useState('suggest');
    const [strategy, setStrategy]       = useState('combined');
    const [budget, setBudget]           = useState(10);
    const [topK, setTopK]               = useState(0);
    const [modelType, setModelType]     = useState('seed');
    const [highConf, setHighConf]       = useState(0.60);
    const [lowConf, setLowConf]         = useState(0.25);
    const [useTta, setUseTta]           = useState(false);
    const [jobs, setJobs]               = useState([]);
    const [activeJobId, setActiveJobId] = useState(null);
    const [launching, setLaunching]     = useState(false);
    const [copied, setCopied]           = useState(false);

    const pollRef = useRef({});
    const jobsRef = useRef(jobs);
    const logsEnd = useRef(null);

    useEffect(() => { jobsRef.current = jobs; }, [jobs]);

    // ── Load model status ─────────────────────────────────────
    const loadSetup = useCallback(() => {
        axios.get(`${API_URL}/pipeline/model-status/${project.id}`)
            .then(res => setModelStatus(res.data))
            .catch(() => {});
    }, [project.id]);

    // ── Load persisted jobs from DB on mount ──────────────────
    const loadPersistedJobs = useCallback((resumePollingFn) => {
        axios.get(`${API_URL}/pipeline/jobs/${project.id}?job_type=active_learning`)
            .then(async res => {
                if (!res.data.length) return;
                const DB_STATUS = { pending: 'PENDING', started: 'STARTED', success: 'SUCCESS', failure: 'FAILURE' };
                const loaded = res.data.map(j => ({
                    id:        j.id,
                    taskId:    j.id,
                    mode:      j.result_meta?.mode   || 'suggest',
                    config:    j.result_meta?.config || {},
                    status:    DB_STATUS[j.status]   || 'PENDING',
                    logs:      j.result_meta?.logs   || [`[ID] Task ID: ${j.id}`],
                    progress:  j.result_meta?.progress || null,
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
                            const newLogs = [...job.logs, '✓ Done! (completed while panel was closed)'];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'success',
                                result_meta: { mode: job.mode, config: job.config, logs: newLogs, result },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            if (onAnnotationsUpdated && job.mode === 'curriculum') onAnnotationsUpdated();
                            return { ...job, status: 'SUCCESS', result, logs: newLogs };

                        } else if (celery === 'FAILURE') {
                            const newLogs = [...job.logs, `Failed: ${error || 'Unknown error'}`];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { mode: job.mode, config: job.config, logs: newLogs, error },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...job, status: 'FAILURE', error, logs: newLogs };

                        } else if (celery === 'PENDING' && job.status === 'STARTED') {
                            const newLogs = [...job.logs, 'Task result expired (panel was closed before it finished).'];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { mode: job.mode, config: job.config, logs: newLogs },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...job, status: 'FAILURE', logs: newLogs };

                        } else {
                            resumePollingFn(job.taskId, job.mode);
                            return job;
                        }
                    } catch {
                        resumePollingFn(job.taskId, job.mode);
                        return job;
                    }
                }));

                setJobs(reconciled);
                setActiveJobId(reconciled[reconciled.length - 1].id);
            })
            .catch(() => {});
    }, [project.id, onAnnotationsUpdated]);

    useEffect(() => {
        loadSetup();
        loadPersistedJobs(startPolling);
        const polls = pollRef.current;
        return () => { Object.values(polls).forEach(clearInterval); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [loadSetup, loadPersistedJobs]);

    useEffect(() => {
        if (logsEnd.current) logsEnd.current.scrollIntoView({ behavior: 'smooth' });
    }, [jobs, activeJobId]);

    // ── Polling ───────────────────────────────────────────────
    const startPolling = useCallback((taskId, jobMode) => {
        if (pollRef.current[taskId]) clearInterval(pollRef.current[taskId]);
        let pendingTicks     = 0;
        let startedPersisted = false;

        pollRef.current[taskId] = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/pipeline/task-status/${taskId}`);
                const { status, result, meta, error } = res.data;

                let isNoWorker = false;
                if (status === 'PENDING') {
                    pendingTicks++;
                    if (pendingTicks >= NO_WORKER_TICKS) isNoWorker = true;
                } else if (status === 'STARTED') {
                    pendingTicks = 0;
                }

                // All side effects BEFORE setJobs
                if (status === 'SUCCESS' || status === 'FAILURE' || isNoWorker) {
                    clearInterval(pollRef.current[taskId]);
                    delete pollRef.current[taskId];
                }

                if (status === 'STARTED' && !startedPersisted) {
                    startedPersisted = true;
                    axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, { status: 'started' }).catch(() => {});
                }

                if (status === 'SUCCESS') {
                    const j = jobsRef.current.find(j => j.taskId === taskId);
                    if (j) {
                        const successLogs = [...j.logs, '✓ Done!'];
                        axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                            status: 'success',
                            result_meta: {
                                mode: j.mode, config: j.config, logs: successLogs, result,
                                startedAt: j.startedAt instanceof Date ? j.startedAt.toISOString() : j.startedAt,
                            },
                            finished_at: new Date().toISOString(),
                        }).catch(() => {});
                    }
                    if (onAnnotationsUpdated && jobMode === 'curriculum') onAnnotationsUpdated();
                    loadSetup();
                }

                if (status === 'FAILURE') {
                    const j = jobsRef.current.find(j => j.taskId === taskId);
                    if (j) {
                        const failLogs = [...j.logs, `Failed: ${error || 'Unknown error'}`];
                        axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                            status: 'failure',
                            result_meta: {
                                mode: j.mode, config: j.config, logs: failLogs, error,
                                startedAt: j.startedAt instanceof Date ? j.startedAt.toISOString() : j.startedAt,
                            },
                            finished_at: new Date().toISOString(),
                        }).catch(() => {});
                    }
                }

                setJobs(prev => prev.map(j => {
                    if (j.taskId !== taskId) return j;
                    let newLogs = [...j.logs];

                    if (status === 'PENDING') {
                        if (isNoWorker) {
                            return { ...j, status: 'NO_WORKER', logs: [...newLogs, 'No worker detected after 37s.', 'Start Celery worker first.'] };
                        }
                        const msg  = `Waiting for worker… (${pendingTicks * 2}s)`;
                        const last = newLogs[newLogs.length - 1];
                        newLogs = last?.startsWith('Waiting') ? [...newLogs.slice(0, -1), msg] : [...newLogs, msg];
                        return { ...j, status: 'PENDING', logs: newLogs };
                    }

                    if (status === 'STARTED') {
                        const progress = meta || j.progress;
                        if (meta?.current) {
                            const entry = `Scoring [${meta.current}/${meta.total}]…`;
                            const last  = newLogs[newLogs.length - 1];
                            newLogs = last?.startsWith('Scoring') ? [...newLogs.slice(0, -1), entry] : [...newLogs, entry];
                        }
                        return { ...j, status: 'STARTED', progress, logs: newLogs };
                    }

                    if (status === 'SUCCESS') {
                        return { ...j, status: 'SUCCESS', result, logs: [...newLogs, '✓ Done!'] };
                    }

                    if (status === 'FAILURE') {
                        return { ...j, status: 'FAILURE', error, logs: [...newLogs, `Failed: ${error || 'Unknown error'}`] };
                    }

                    return j;
                }));
            } catch { /* ignore */ }
        }, POLL_INTERVAL);
    }, [loadSetup, onAnnotationsUpdated]);

    // ── Stop running job ──────────────────────────────────────
    const handleStop = async () => {
        const job = jobs.find(j => j.status === 'PENDING' || j.status === 'STARTED');
        if (!job?.taskId) return;
        try { await axios.post(`${API_URL}/pipeline/cancel/${job.taskId}`); } catch { /* ignore */ }
        if (pollRef.current[job.taskId]) {
            clearInterval(pollRef.current[job.taskId]);
            delete pollRef.current[job.taskId];
        }
        setJobs(prev => prev.map(j =>
            j.taskId === job.taskId
                ? { ...j, status: 'FAILURE', logs: [...j.logs, 'Stopped by user.'] }
                : j
        ));
    };

    // ── Copy celery command ───────────────────────────────────
    const handleCopy = () => {
        navigator.clipboard.writeText('celery -A app.tasks.celery_app:celery_app worker --loglevel=info');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    // ── Launch job ────────────────────────────────────────────
    const handleStart = async () => {
        setLaunching(true);
        setView('jobs');

        let endpoint = '';
        let body     = {};
        const jobConfig = { mode, strategy };

        if (mode === 'suggest') {
            endpoint = `${API_URL}/pipeline/active-learning/suggest/${project.id}`;
            body = { budget, strategy };
            jobConfig.budget = budget;
        } else if (mode === 'curriculum') {
            endpoint = `${API_URL}/pipeline/active-learning/curriculum-annotate/${project.id}`;
            body = { high_conf: highConf, low_conf: lowConf, use_tta: useTta };
            jobConfig.high_conf = highConf;
            jobConfig.low_conf  = lowConf;
        } else if (mode === 'score') {
            endpoint = `${API_URL}/pipeline/active-learning/score/${project.id}`;
            body = { model_type: modelType, strategy, top_k: topK, use_tta: useTta };
            jobConfig.top_k      = topK;
            jobConfig.model_type = modelType;
        }

        try {
            const res    = await axios.post(endpoint, body);
            const taskId = res.data.task_id;
            const job    = makeJob(taskId, mode, jobConfig);
            setJobs(prev => [...prev, job]);
            setActiveJobId(job.id);
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: taskId, project_id: project.id, job_type: 'active_learning',
                result_meta: { mode, config: jobConfig, logs: job.logs, startedAt: job.startedAt.toISOString() },
            }).catch(() => {});
            startPolling(taskId, mode);
        } catch {
            const failJob = {
                id: Date.now(), taskId: null, mode, config: jobConfig,
                status: 'FAILURE', logs: ['Failed to launch job.'],
                progress: null, result: null, error: 'Launch failed', startedAt: new Date(),
            };
            setJobs(prev => [...prev, failJob]);
            setActiveJobId(failJob.id);
        } finally {
            setLaunching(false);
        }
    };

    // ── Derived ───────────────────────────────────────────────
    const activeJob   = jobs.find(j => j.id === activeJobId) || jobs[jobs.length - 1] || null;
    const anyRunning  = jobs.some(j => j.status === 'PENDING' || j.status === 'STARTED');
    const hasSeed     = modelStatus?.has_seed_model;
    const hasMain     = modelStatus?.has_main_model;
    const modelNeeded = mode === 'score' ? (modelType === 'seed' ? hasSeed : hasMain) : hasSeed;
    const canStart    = modelNeeded && !launching;

    return (
        <div className="alp-overlay" onClick={onClose}>
            <div className="alp-panel" onClick={e => e.stopPropagation()}>

                {/* ── Header ── */}
                <div className="alp-header">
                    <div className="alp-header-left">
                        <div>
                            <h2 className="alp-title">Active Learning</h2>
                            <p className="alp-subtitle">{project.name}</p>
                        </div>
                    </div>
                    <button className="alp-close" onClick={onClose}><X size={16} /></button>
                </div>

                {/* ── Tabs ── */}
                <div className="alp-tabs">
                    <button className={`alp-tab ${view === 'setup' ? 'alp-tab--active' : ''}`} onClick={() => setView('setup')}>
                        Setup
                    </button>
                    <button className={`alp-tab ${view === 'jobs' ? 'alp-tab--active' : ''}`} onClick={() => setView('jobs')}>
                        Jobs
                        {jobs.length > 0 && <span className="alp-tab-badge">{jobs.length}</span>}
                        {anyRunning && <span className="alp-tab-dot" />}
                    </button>
                </div>

                <div className="alp-body">

                    {/* ═══════════ SETUP ═══════════ */}
                    {view === 'setup' && (
                        <>
                            {/* Model status */}
                            <section className="alp-section">
                                <p className="alp-section-title">Model Status</p>
                                {modelStatus == null ? (
                                    <div className="alp-loading"><div className="alp-spinner" /><span>Checking…</span></div>
                                ) : (
                                    <div className="alp-model-row">
                                        <div className={`alp-model-chip ${hasSeed ? 'alp-model-chip--ok' : 'alp-model-chip--miss'}`}>
                                            <Circle size={8} className={hasSeed ? 'alp-dot-green' : 'alp-dot-red'} fill="currentColor" />
                                            Seed Model
                                        </div>
                                        <div className={`alp-model-chip ${hasMain ? 'alp-model-chip--ok' : 'alp-model-chip--miss'}`}>
                                            <Circle size={8} className={hasMain ? 'alp-dot-green' : 'alp-dot-red'} fill="currentColor" />
                                            Main Model
                                        </div>
                                        <button className="alp-refresh-btn" onClick={loadSetup} title="Refresh">
                                            <RefreshCw size={14} />
                                        </button>
                                    </div>
                                )}
                                {!hasSeed && modelStatus != null && (
                                    <p className="alp-hint" style={{ marginTop: 8 }}>
                                        Train the seed model first using <strong>Train Seed Model</strong>.
                                    </p>
                                )}
                            </section>

                            {/* Mode selection */}
                            <section className="alp-section">
                                <p className="alp-section-title">Mode</p>
                                <div className="alp-mode-grid">
                                    {MODES.map(({ id, Icon, label, desc }) => (
                                        <div
                                            key={id}
                                            className={`alp-mode-card ${mode === id ? 'alp-mode-card--active' : ''}`}
                                            onClick={() => setMode(id)}
                                        >
                                            <span className="alp-mode-icon"><Icon size={16} /></span>
                                            <div>
                                                <span className="alp-mode-label">{label}</span>
                                                <span className="alp-mode-desc">{desc}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </section>

                            {/* ── Suggest config ── */}
                            {mode === 'suggest' && (
                                <section className="alp-section">
                                    <p className="alp-section-title">Configuration</p>
                                    <div className="alp-field">
                                        <label className="alp-label">
                                            Images to Suggest
                                            <span className="alp-hint-text">Number of most uncertain images to return</span>
                                        </label>
                                        <div className="alp-conf-control">
                                            <input
                                                type="range" min="1" max="50" step="1"
                                                value={budget}
                                                onChange={e => setBudget(parseInt(e.target.value))}
                                                className="alp-conf-slider"
                                            />
                                            <span className="alp-conf-val">{budget}</span>
                                        </div>
                                    </div>
                                    <div className="alp-field" style={{ marginTop: 14 }}>
                                        <label className="alp-label">Uncertainty Strategy</label>
                                        <div className="alp-strategy-grid">
                                            {STRATEGIES.map(s => (
                                                <div
                                                    key={s.id}
                                                    className={`alp-strategy-card ${strategy === s.id ? 'alp-strategy-card--active' : ''}`}
                                                    onClick={() => setStrategy(s.id)}
                                                >
                                                    <span className="alp-strategy-label">{s.label}</span>
                                                    <span className="alp-strategy-desc">{s.desc}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </section>
                            )}

                            {/* ── Curriculum config ── */}
                            {mode === 'curriculum' && (
                                <section className="alp-section">
                                    <p className="alp-section-title">Confidence Thresholds</p>
                                    <div className="alp-field">
                                        <label className="alp-label">
                                            Auto-Accept threshold
                                            <span className="alp-hint-text">Confidence ≥ this → annotated automatically</span>
                                        </label>
                                        <div className="alp-conf-control">
                                            <input
                                                type="range" min="0.40" max="0.95" step="0.05"
                                                value={highConf}
                                                onChange={e => setHighConf(parseFloat(e.target.value))}
                                                className="alp-conf-slider"
                                            />
                                            <span className="alp-conf-val">{fmtPct(highConf)}</span>
                                        </div>
                                    </div>
                                    <div className="alp-field" style={{ marginTop: 14 }}>
                                        <label className="alp-label">
                                            Review band lower bound
                                            <span className="alp-hint-text">Confidence between this and auto-accept → saved for review</span>
                                        </label>
                                        <div className="alp-conf-control">
                                            <input
                                                type="range" min="0.10" max="0.55" step="0.05"
                                                value={lowConf}
                                                onChange={e => setLowConf(parseFloat(e.target.value))}
                                                className="alp-conf-slider"
                                            />
                                            <span className="alp-conf-val">{fmtPct(lowConf)}</span>
                                        </div>
                                    </div>
                                    <div className="alp-tier-visual">
                                        <div className="alp-tier alp-tier--auto">≥{fmtPct(highConf)} → Auto-Accept</div>
                                        <div className="alp-tier alp-tier--review">{fmtPct(lowConf)}–{fmtPct(highConf)} → Review</div>
                                        <div className="alp-tier alp-tier--skip">&lt;{fmtPct(lowConf)} → Skip (annotate manually)</div>
                                    </div>
                                    <label className="alp-checkbox-row">
                                        <input type="checkbox" checked={useTta} onChange={e => setUseTta(e.target.checked)} />
                                        <span>Use TTA for more robust predictions (slower)</span>
                                    </label>
                                </section>
                            )}

                            {/* ── Score config ── */}
                            {mode === 'score' && (
                                <section className="alp-section">
                                    <p className="alp-section-title">Configuration</p>
                                    <div className="alp-field">
                                        <label className="alp-label">Model to use for scoring</label>
                                        <div className="alp-model-type-row">
                                            {['seed', 'main'].map(mt => (
                                                <button
                                                    key={mt}
                                                    className={`alp-model-type-btn ${modelType === mt ? 'alp-model-type-btn--active' : ''}`}
                                                    onClick={() => setModelType(mt)}
                                                >
                                                    {mt === 'seed' ? 'Seed Model' : 'Main Model'}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="alp-field" style={{ marginTop: 14 }}>
                                        <label className="alp-label">Uncertainty Strategy</label>
                                        <div className="alp-strategy-grid">
                                            {STRATEGIES.map(s => (
                                                <div
                                                    key={s.id}
                                                    className={`alp-strategy-card ${strategy === s.id ? 'alp-strategy-card--active' : ''}`}
                                                    onClick={() => setStrategy(s.id)}
                                                >
                                                    <span className="alp-strategy-label">{s.label}</span>
                                                    <span className="alp-strategy-desc">{s.desc}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="alp-field" style={{ marginTop: 14 }}>
                                        <label className="alp-label">
                                            Top-K images
                                            <span className="alp-hint-text">0 = return all pending images ranked</span>
                                        </label>
                                        <div className="alp-conf-control">
                                            <input
                                                type="range" min="0" max="100" step="5"
                                                value={topK}
                                                onChange={e => setTopK(parseInt(e.target.value))}
                                                className="alp-conf-slider"
                                            />
                                            <span className="alp-conf-val">{topK === 0 ? 'All' : topK}</span>
                                        </div>
                                    </div>
                                    <label className="alp-checkbox-row" style={{ marginTop: 10 }}>
                                        <input type="checkbox" checked={useTta} onChange={e => setUseTta(e.target.checked)} />
                                        <span>Include TTA disagreement scoring</span>
                                    </label>
                                </section>
                            )}

                            {/* Worker note */}
                            <section className="alp-section">
                                <p className="alp-section-title">Worker Required</p>
                                <div className="alp-cmd-block">
                                    <code>celery -A app.tasks.celery_app:celery_app worker --loglevel=info</code>
                                    <button className="alp-cmd-copy" onClick={handleCopy} title="Copy">
                                        {copied ? <Check size={13} /> : <Clipboard size={13} />}
                                    </button>
                                </div>
                            </section>
                        </>
                    )}

                    {/* ═══════════ JOBS ═══════════ */}
                    {view === 'jobs' && (
                        jobs.length === 0 ? (
                            <div className="alp-jobs-empty">
                                <Inbox size={32} />
                                <p>No jobs yet. Configure settings in Setup and click Run.</p>
                            </div>
                        ) : (
                            <div className="alp-jobs-layout">
                                {/* Job list */}
                                <div className="alp-jobs-list">
                                    {[...jobs].reverse().map((job, idx) => {
                                        const info     = STATUS_LABEL[job.status] || { label: job.status, cls: 'alp-badge--pending' };
                                        const modeInfo = MODES.find(m => m.id === job.mode);
                                        return (
                                            <div
                                                key={job.id}
                                                className={`alp-job-item ${activeJob?.id === job.id ? 'alp-job-item--active' : ''}`}
                                                onClick={() => setActiveJobId(job.id)}
                                            >
                                                <div className="alp-job-item-top">
                                                    <span className="alp-job-num">#{jobs.length - idx}</span>
                                                    <span className={`alp-job-badge ${info.cls}`}>{info.label}</span>
                                                </div>
                                                <div className="alp-job-meta">
                                                    {modeInfo && <modeInfo.Icon size={10} style={{ display: 'inline', marginRight: 3 }} />}
                                                    {modeInfo?.label}
                                                </div>
                                                <div className="alp-job-time">{fmtTime(job.startedAt)}</div>
                                            </div>
                                        );
                                    })}
                                </div>

                                {/* Job detail */}
                                {activeJob && (
                                    <div className="alp-job-detail">
                                        <div className="alp-job-detail-header">
                                            {(() => {
                                                const info = STATUS_LABEL[activeJob.status] || { label: activeJob.status, cls: 'alp-badge--pending' };
                                                return <span className={`alp-job-badge alp-job-badge--lg ${info.cls}`}>{info.label}</span>;
                                            })()}
                                            <span className="alp-job-detail-time">{fmtTime(activeJob.startedAt)}</span>
                                            {(activeJob.status === 'PENDING' || activeJob.status === 'STARTED') && (
                                                <span className="alp-live-badge"><Circle size={6} fill="currentColor" /> Live</span>
                                            )}
                                        </div>

                                        {activeJob.taskId && (
                                            <div className="alp-taskid">
                                                <span className="alp-taskid-label">Task ID</span>
                                                <span className="alp-taskid-val">{activeJob.taskId}</span>
                                            </div>
                                        )}

                                        {/* Progress bar */}
                                        <ProgressBar progress={activeJob.progress} />

                                        {/* Log */}
                                        <div className="alp-section-label" style={{ marginBottom: 6 }}>Log</div>
                                        <div className="alp-logs">
                                            {activeJob.logs.map((line, i) => (
                                                <div key={i} className="alp-log-line">{line}</div>
                                            ))}
                                            <div ref={logsEnd} />
                                        </div>

                                        {/* Results */}
                                        {activeJob.status === 'SUCCESS' && activeJob.result && (
                                            <>
                                                {activeJob.mode === 'curriculum' && (
                                                    <CurriculumResult result={activeJob.result} />
                                                )}
                                                {activeJob.mode === 'suggest' && (
                                                    <ScoreResultTable
                                                        items={activeJob.result.suggestions?.map((s, i) => ({
                                                            ...s, rank: i + 1, combined_score: s.uncertainty,
                                                        }))}
                                                        label="Suggested Images to Annotate Manually"
                                                        onAnnotate={onAnnotateImages ? (ids) => {
                                                            onAnnotateImages(ids);
                                                            onClose();
                                                        } : undefined}
                                                    />
                                                )}
                                                {activeJob.mode === 'score' && (
                                                    <ScoreResultTable
                                                        items={activeJob.result.scored_images}
                                                        label="Images Ranked by Uncertainty"
                                                        onAnnotate={onAnnotateImages ? (ids) => {
                                                            onAnnotateImages(ids);
                                                            onClose();
                                                        } : undefined}
                                                    />
                                                )}
                                            </>
                                        )}

                                        {/* No worker */}
                                        {activeJob.status === 'NO_WORKER' && (
                                            <div className="alp-worker-err">
                                                <div className="alp-worker-err-title">
                                                    <AlertTriangle size={14} />
                                                    Worker not found. Start it then try again:
                                                </div>
                                                <div className="alp-cmd-block" style={{ marginTop: 8 }}>
                                                    <code>celery -A app.tasks.celery_app:celery_app worker --loglevel=info</code>
                                                    <button className="alp-cmd-copy" onClick={handleCopy}>
                                                        {copied ? <Check size={13} /> : <Clipboard size={13} />}
                                                    </button>
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
                <div className="alp-footer">
                    {view === 'setup' ? (
                        <button className="alp-start-btn" onClick={handleStart} disabled={!canStart}>
                            {launching ? (
                                <><div className="alp-btn-spinner" /> Starting…</>
                            ) : mode === 'suggest' ? (
                                <><Target size={15} /> Suggest {budget} Image{budget !== 1 ? 's' : ''}</>
                            ) : mode === 'curriculum' ? (
                                <><Zap size={15} /> Run Curriculum Annotate</>
                            ) : (
                                <><Play size={15} /> Score Images</>
                            )}
                        </button>
                    ) : (
                        <button className="alp-start-btn alp-start-btn--secondary" onClick={() => setView('setup')}>
                            <ChevronLeft size={15} /> Back to Setup
                        </button>
                    )}
                    {anyRunning && (
                        <button className="alp-stop-btn" onClick={handleStop}>
                            <Square size={13} /> Stop
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ActiveLearningPanel;
