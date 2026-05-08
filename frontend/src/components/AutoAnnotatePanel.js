import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './AutoAnnotatePanel.css';
import { Sparkles, X, RefreshCw, Square, AlertTriangle, Check, ChevronLeft, Clock, Clipboard } from 'lucide-react';
import logoImg from '../logo.png';

import { API_URL, BASE_URL } from '../config';
const IMG_URL  = BASE_URL;
const POLL_INTERVAL  = 2500;
const NO_WORKER_TICKS = 15; // ~37s

const STATUS_LABEL = {
    QUEUED:    { label: 'Queued',    cls: 'aap-badge--queued'  },
    PENDING:   { label: 'Pending',   cls: 'aap-badge--pending' },
    STARTED:   { label: 'Running',   cls: 'aap-badge--running' },
    SUCCESS:   { label: 'Done',      cls: 'aap-badge--done'    },
    FAILURE:   { label: 'Failed',    cls: 'aap-badge--fail'    },
    NO_WORKER: { label: 'No Worker', cls: 'aap-badge--fail'    },
};

const fmtTime = (d) => new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

const makeJob = (taskId, count) => ({
    id: Date.now(),
    taskId,
    imageCount: count,
    status: 'PENDING',
    logs: [`[ID] Task ID: ${taskId}`, `[IMG] Processing ${count} image${count !== 1 ? 's' : ''}...`],
    progress: null,  // { current, total, current_image, annotated_count }
    result: null,
    error: null,
    startedAt: new Date(),
});

// ── ImageCheckbox ────────────────────────────────────────────────
const ImageThumb = ({ img, checked, onChange }) => (
    <label className={`aap-img-thumb ${checked ? 'aap-img-thumb--checked' : ''}`}>
        <input type="checkbox" checked={checked} onChange={onChange} className="aap-img-checkbox" />
        <div className="aap-img-preview">
            <img
                src={`${IMG_URL}${img.filepath}`}
                alt={img.filename}
                onError={e => { e.target.style.display = 'none'; }}
            />
        </div>
        <span className="aap-img-name">{img.filename}</span>
        {checked && <span className="aap-img-check"><Check size={12} /></span>}
    </label>
);

// ── Progress Bar ────────────────────────────────────────────────
const ProgressBar = ({ progress }) => {
    if (!progress || !progress.total) return null;
    const pct = Math.round((progress.current / progress.total) * 100);
    const skipped  = progress.skipped_path  || 0;
    const noDetect = progress.no_detection  || 0;
    return (
        <div className="aap-progress">
            <div className="aap-progress-header">
                <span className="aap-progress-label">
                    Image <strong>{progress.current}</strong> / {progress.total}
                </span>
                {progress.current_image && (
                    <span className="aap-progress-file">{progress.current_image}</span>
                )}
                <span className="aap-progress-pct">{pct}%</span>
            </div>
            <div className="aap-progress-bar">
                <div className="aap-progress-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="aap-progress-stats">
                {(progress.annotated_count > 0) && (
                    <span className="aap-stat aap-stat--ok">✅ {progress.annotated_count} annotated</span>
                )}
                {noDetect > 0 && (
                    <span className="aap-stat aap-stat--warn">⬜ {noDetect} no detection</span>
                )}
                {skipped > 0 && (
                    <span className="aap-stat aap-stat--err"><AlertTriangle size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 2 }} /> {skipped} file not found</span>
                )}
            </div>
        </div>
    );
};

// ════════════════════════════════════════════════════════════
//  AutoAnnotatePanel
// ════════════════════════════════════════════════════════════
const AutoAnnotatePanel = ({ project, onClose, onAnnotationsUpdated }) => {
    const [view, setView]               = useState('setup');
    const [modelStatus, setModelStatus] = useState(null);
    const [pendingImages, setPendingImages] = useState([]);
    const [loadingImages, setLoadingImages] = useState(true);
    const [selectedIds, setSelectedIds] = useState(new Set());
    const [conf, setConf]               = useState(0.10);
    const [jobs, setJobs]               = useState([]);
    const [activeJobId, setActiveJobId] = useState(null);
    const [launching, setLaunching]     = useState(false);

    const pollRef  = useRef({});
    const jobsRef  = useRef(jobs);
    const logsEnd  = useRef(null);

    useEffect(() => { jobsRef.current = jobs; }, [jobs]);

    // ── Load pending images + model status ───────────────────
    const loadSetup = useCallback(() => {
        setLoadingImages(true);
        Promise.all([
            axios.get(`${API_URL}/pipeline/pending-images/${project.id}`),
            axios.get(`${API_URL}/pipeline/model-status/${project.id}`),
        ])
            .then(([imgRes, modelRes]) => {
                setPendingImages(imgRes.data);
                setModelStatus(modelRes.data);
                // Default: select all
                setSelectedIds(new Set(imgRes.data.map(i => i.id)));
            })
            .catch(() => {})
            .finally(() => setLoadingImages(false));
    }, [project.id]);

    // ── Load persisted jobs from DB on mount ──────────
    const loadPersistedJobs = useCallback((resumePollingFn) => {
        axios.get(`${API_URL}/pipeline/jobs/${project.id}?job_type=auto_annotate`)
            .then(async res => {
                if (!res.data.length) return;
                const DB_STATUS = { pending: 'PENDING', started: 'STARTED', success: 'SUCCESS', failure: 'FAILURE' };
                const loaded = res.data.map(j => ({
                    id:         j.id,
                    taskId:     j.id,
                    imageCount: j.result_meta?.imageCount || 0,
                    status:     DB_STATUS[j.status] || 'PENDING',
                    logs:       j.result_meta?.logs     || [`📋  Task ID: ${j.id}`],
                    progress:   j.result_meta?.progress || null,
                    result:     j.result_meta?.result   || null,
                    error:      j.result_meta?.error    || null,
                    startedAt:  new Date(j.created_at),
                }));

                // ── Reconcile ALL active jobs in parallel BEFORE rendering ──
                // Using Promise.all means we only call setJobs ONCE, with the
                // final reconciled statuses — eliminating the flicker that occurred
                // when setJobs was called first with DB status then again after check.
                const reconciled = await Promise.all(loaded.map(async (job) => {
                    if (job.status !== 'PENDING' && job.status !== 'STARTED') return job;

                    try {
                        const sr = await axios.get(`${API_URL}/pipeline/task-status/${job.taskId}`);
                        const { status: celery, result, error } = sr.data;

                        if (celery === 'SUCCESS') {
                            const newLogs = [...job.logs, '✅  Done! (completed while panel was closed)'];
                            const finalProgress = {
                                current: job.imageCount, total: job.imageCount,
                                annotated_count: result?.annotated_count,
                            };
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'success',
                                result_meta: { logs: newLogs, result, progress: finalProgress, imageCount: job.imageCount },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            loadSetup();
                            if (onAnnotationsUpdated) onAnnotationsUpdated();
                            return { ...job, status: 'SUCCESS', result, logs: newLogs, progress: finalProgress };

                        } else if (celery === 'FAILURE') {
                            const newLogs = [...job.logs, `❌  Failed: ${error || 'Unknown error'}`];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { logs: newLogs, error, imageCount: job.imageCount },
                                finished_at: new Date().toISOString(),
                            }).catch(() => {});
                            return { ...job, status: 'FAILURE', error, logs: newLogs };

                        } else if (celery === 'PENDING' && job.status === 'STARTED') {
                            // DB says "started" but Celery shows PENDING — result expired.
                            const newLogs = [...job.logs,
                                '⚠️  Task result expired (panel was closed before it finished).',
                                '    Check your annotations — images may already be annotated.',
                            ];
                            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                                status: 'failure',
                                result_meta: { logs: newLogs, imageCount: job.imageCount },
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
    }, [project.id, loadSetup, onAnnotationsUpdated]);

    useEffect(() => {
        loadSetup();
        loadPersistedJobs(startPolling);
        const polls = pollRef.current;
        return () => { Object.values(polls).forEach(clearInterval); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [loadSetup, loadPersistedJobs]);

    // auto-scroll logs
    useEffect(() => {
        if (logsEnd.current) logsEnd.current.scrollIntoView({ behavior: 'smooth' });
    }, [jobs, activeJobId]);

    // ── Toggle selection ─────────────────────────────────────
    const toggleImage = (id) => setSelectedIds(prev => {
        const next = new Set(prev);
        next.has(id) ? next.delete(id) : next.add(id);
        return next;
    });

    const selectAll   = () => setSelectedIds(new Set(pendingImages.map(i => i.id)));
    const selectNone  = () => setSelectedIds(new Set());

    // ── Polling ───────────────────────────────────────────────
    const startPolling = useCallback((taskId) => {
        if (pollRef.current[taskId]) clearInterval(pollRef.current[taskId]);
        let pendingTicks = 0;
        let startedPersisted = false;

        pollRef.current[taskId] = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/pipeline/task-status/${taskId}`);
                const { status, result, meta, error } = res.data;

                // ── Compute NO_WORKER threshold outside setJobs ───────
                let isNoWorker = false;
                if (status === 'PENDING') {
                    pendingTicks++;
                    if (pendingTicks >= NO_WORKER_TICKS) isNoWorker = true;
                } else if (status === 'STARTED') {
                    pendingTicks = 0;
                }

                // ── All side effects BEFORE setJobs (pure updater) ────
                // Stopping the interval and firing callbacks must not happen
                // inside the setJobs functional updater — React may call it
                // multiple times, which would double-trigger loadSetup() and
                // cause the image grid to blank out unexpectedly.
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
                        const successLogs = [...j.logs, `✅  Done! ${result?.annotated_count ?? 0} image(s) annotated.`];
                        const finalProgress = { current: j.imageCount, total: j.imageCount, annotated_count: result?.annotated_count };
                        axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                            status: 'success',
                            result_meta: {
                                logs: successLogs, result, progress: finalProgress,
                                imageCount: j.imageCount,
                                startedAt: j.startedAt instanceof Date ? j.startedAt.toISOString() : j.startedAt,
                            },
                            finished_at: new Date().toISOString(),
                        }).catch(() => {});
                    }
                    loadSetup();
                    if (onAnnotationsUpdated) onAnnotationsUpdated();
                }

                if (status === 'FAILURE') {
                    const j = jobsRef.current.find(j => j.taskId === taskId);
                    if (j) {
                        const failLogs = [...j.logs, `❌  Failed: ${error || 'Unknown error'}`];
                        axios.patch(`${API_URL}/pipeline/jobs/${taskId}`, {
                            status: 'failure',
                            result_meta: {
                                logs: failLogs, error: error || 'Unknown error',
                                imageCount: j.imageCount,
                                startedAt: j.startedAt instanceof Date ? j.startedAt.toISOString() : j.startedAt,
                            },
                            finished_at: new Date().toISOString(),
                        }).catch(() => {});
                    }
                }

                // ── Pure state update — no side effects inside ────────
                setJobs(prev => prev.map(j => {
                    if (j.taskId !== taskId) return j;
                    let newLogs = [...j.logs];

                    if (status === 'PENDING') {
                        if (isNoWorker) {
                            newLogs = [...newLogs, '❌  No worker detected after 37s.', '👉  Start Celery worker first.'];
                            return { ...j, status: 'NO_WORKER', logs: newLogs };
                        }
                        const msg = `⏳  Waiting for worker… (${pendingTicks * 2}s)`;
                        const last = newLogs[newLogs.length - 1];
                        newLogs = last?.startsWith('⏳') ? [...newLogs.slice(0, -1), msg] : [...newLogs, msg];
                        return { ...j, status: 'PENDING', logs: newLogs };
                    }

                    if (status === 'STARTED') {
                        const progress = meta || j.progress;
                        if (meta?.current_image) {
                            const entry = `🔍  [${meta.current}/${meta.total}] ${meta.current_image}`;
                            const last = newLogs[newLogs.length - 1];
                            if (last !== entry) newLogs = [...newLogs, entry];
                        }
                        return { ...j, status: 'STARTED', progress, logs: newLogs };
                    }

                    if (status === 'SUCCESS') {
                        newLogs = [...newLogs, `✅  Done! ${result?.annotated_count ?? 0} image(s) annotated.`];
                        const finalProgress = { current: j.imageCount, total: j.imageCount, annotated_count: result?.annotated_count };
                        return { ...j, status: 'SUCCESS', result, logs: newLogs, progress: finalProgress };
                    }

                    if (status === 'FAILURE') {
                        newLogs = [...newLogs, `❌  Failed: ${error || 'Unknown error'}`];
                        return { ...j, status: 'FAILURE', error, logs: newLogs };
                    }

                    return j;
                }));
            } catch { /* ignore */ }
        }, POLL_INTERVAL);
    }, [loadSetup, onAnnotationsUpdated]);

    // ── Stop running job ─────────────────────────────────────────
    const handleStop = async () => {
        const job = jobs.find(j => j.status === 'PENDING' || j.status === 'STARTED');
        if (!job?.taskId) return;
        try {
            await axios.post(`${API_URL}/pipeline/cancel/${job.taskId}`);
        } catch { /* ignore */ }
        if (pollRef.current[job.taskId]) {
            clearInterval(pollRef.current[job.taskId]);
            delete pollRef.current[job.taskId];
        }
        setJobs(prev => prev.map(j =>
            j.taskId === job.taskId
                ? { ...j, status: 'FAILURE', logs: [...j.logs, '🛑  Stopped by user.'] }
                : j
        ));
    };

    // ── Launch job ────────────────────────────────────────────
    const handleStart = async () => {
        if (!modelStatus?.has_seed_model) return;
        if (selectedIds.size === 0) return;
        setLaunching(true);
        setView('jobs');

        const ids = [...selectedIds];
        try {
            const res = await axios.post(
                `${API_URL}/pipeline/auto-annotate/${project.id}`,
                { image_ids: ids, conf }
            );
            const taskId = res.data.task_id;
            const job = makeJob(taskId, ids.length);
            setJobs(prev => [...prev, job]);
            setActiveJobId(job.id);
            // Persist to DB so jobs survive page reload
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: taskId, project_id: project.id, job_type: 'auto_annotate',
                conf_used: conf,
                result_meta: { logs: job.logs, imageCount: ids.length, startedAt: job.startedAt.toISOString() },
            }).catch(() => {});
            startPolling(taskId);
        } catch {
            const failJob = {
                id: Date.now(), taskId: null, imageCount: ids.length,
                status: 'FAILURE', logs: ['❌  Failed to launch job.'],
                progress: null, result: null, error: 'Launch failed', startedAt: new Date(),
            };
            setJobs(prev => [...prev, failJob]);
            setActiveJobId(failJob.id);
        } finally {
            setLaunching(false);
        }
    };

    // ── Derived ───────────────────────────────────────────────
    const activeJob  = jobs.find(j => j.id === activeJobId) || jobs[jobs.length - 1] || null;
    const anyRunning = jobs.some(j => j.status === 'PENDING' || j.status === 'STARTED');
    const canStart   = modelStatus?.has_seed_model && selectedIds.size > 0 && !launching;

    return (
        <div className="aap-overlay" onClick={onClose}>
            <div className="aap-panel" onClick={e => e.stopPropagation()}>

                {/* ── Header ── */}
                <div className="aap-header">
                    <div className="aap-header-left">
                        <span className="aap-header-icon"><Sparkles size={20} /></span>
                        <div>
                            <h2 className="aap-title">Auto-Annotate</h2>
                            <p className="aap-subtitle">{project.name}</p>
                        </div>
                    </div>
                    <button className="aap-close" onClick={onClose}><X size={18} /></button>
                </div>

                {/* ── Tabs ── */}
                <div className="aap-tabs">
                    <button className={`aap-tab ${view === 'setup' ? 'aap-tab--active' : ''}`} onClick={() => setView('setup')}>
                        Setup
                    </button>
                    <button className={`aap-tab ${view === 'jobs' ? 'aap-tab--active' : ''}`} onClick={() => setView('jobs')}>
                        Jobs
                        {jobs.length > 0 && <span className="aap-tab-badge">{jobs.length}</span>}
                        {anyRunning && <span className="aap-tab-dot" />}
                    </button>
                </div>

                <div className="aap-body">

                    {/* ═══════════ SETUP ═══════════ */}
                    {view === 'setup' && (
                        <>
                            {/* Model status */}
                            <section className="aap-section">
                                <p className="aap-section-title">Seed Model</p>
                                {modelStatus == null ? (
                                    <div className="aap-loading"><div className="aap-spinner" /><span>Checking…</span></div>
                                ) : modelStatus.has_seed_model ? (
                                    <div className="aap-model-ok">
                                        <span className="aap-model-dot aap-model-dot--green" />
                                        <span>Seed model ready</span>
                                        <span className="aap-model-path">{modelStatus.model_path}</span>
                                    </div>
                                ) : (
                                    <div className="aap-model-missing">
                                        <span className="aap-model-dot aap-model-dot--red" />
                                        <span>No seed model found</span>
                                        <p className="aap-model-hint">Train the seed model first using <strong>Train Seed Model</strong>.</p>
                                    </div>
                                )}
                            </section>

                            {/* Image selection */}
                            <section className="aap-section aap-section--flex">
                                <div className="aap-images-header">
                                    <p className="aap-section-title" style={{ margin: 0 }}>
                                        Pending Images ({pendingImages.length})
                                    </p>
                                    <div className="aap-select-actions">
                                        <button className="aap-select-btn" onClick={selectAll}>All</button>
                                        <button className="aap-select-btn" onClick={selectNone}>None</button>
                                        <span className="aap-selected-count">{selectedIds.size} selected</span>
                                        <button className="aap-refresh-btn" onClick={loadSetup} title="Refresh"><RefreshCw size={16} /></button>
                                    </div>
                                </div>

                                {loadingImages ? (
                                    <div className="aap-loading"><div className="aap-spinner" /><span>Loading images…</span></div>
                                ) : pendingImages.length === 0 ? (
                                    <div className="aap-empty">
                                        <span><Sparkles size={32} /></span>
                                        <p>All images are already annotated!</p>
                                    </div>
                                ) : (
                                    <div className="aap-img-grid">
                                        {pendingImages.map(img => (
                                            <ImageThumb
                                                key={img.id}
                                                img={img}
                                                checked={selectedIds.has(img.id)}
                                                onChange={() => toggleImage(img.id)}
                                            />
                                        ))}
                                    </div>
                                )}
                            </section>

                            {/* Confidence threshold */}
                            <section className="aap-section">
                                <p className="aap-section-title">Detection Settings</p>
                                <div className="aap-conf-row">
                                    <label className="aap-conf-label">
                                        Confidence threshold
                                        <span className="aap-conf-hint">Lower = more detections (may include noise)</span>
                                    </label>
                                    <div className="aap-conf-control">
                                        <input
                                            type="range"
                                            min="0.01" max="0.90" step="0.01"
                                            value={conf}
                                            onChange={e => setConf(parseFloat(e.target.value))}
                                            className="aap-conf-slider"
                                        />
                                        <span className="aap-conf-val">{Math.round(conf * 100)}%</span>
                                    </div>
                                </div>
                            </section>

                            {/* Worker note */}
                            <section className="aap-section">
                                <p className="aap-section-title">Worker Required</p>
                                <div className="aap-cmd-block">
                                    <code>celery -A app.tasks.celery_app:celery_app worker --loglevel=info</code>
                                    <button className="aap-cmd-copy" onClick={() => navigator.clipboard.writeText('celery -A app.tasks.celery_app:celery_app worker --loglevel=info')}><Clipboard size={14} /></button>
                                </div>
                            </section>
                        </>
                    )}

                    {/* ═══════════ JOBS ═══════════ */}
                    {view === 'jobs' && (
                        jobs.length === 0 ? (
                            <div className="aap-jobs-empty">
                                <span><X size={32} /></span>
                                <p>No jobs yet. Select images in Setup and click Start.</p>
                            </div>
                        ) : (
                            <div className="aap-jobs-layout">
                                {/* Job list */}
                                <div className="aap-jobs-list">
                                    {[...jobs].reverse().map((job, idx) => {
                                        const info = STATUS_LABEL[job.status] || { label: job.status, cls: 'aap-badge--pending' };
                                        return (
                                            <div
                                                key={job.id}
                                                className={`aap-job-item ${activeJob?.id === job.id ? 'aap-job-item--active' : ''}`}
                                                onClick={() => setActiveJobId(job.id)}
                                            >
                                                <div className="aap-job-item-top">
                                                    <span className="aap-job-num">#{jobs.length - idx}</span>
                                                    <span className={`aap-job-badge ${info.cls}`}>{info.label}</span>
                                                </div>
                                                <div className="aap-job-meta">{job.imageCount} img</div>
                                                <div className="aap-job-time">{fmtTime(job.startedAt)}</div>
                                            </div>
                                        );
                                    })}
                                </div>

                                {/* Job detail */}
                                {activeJob && (
                                    <div className="aap-job-detail">
                                        {/* Status */}
                                        <div className="aap-job-detail-header">
                                            {(() => {
                                                const info = STATUS_LABEL[activeJob.status] || { label: activeJob.status, cls: 'aap-badge--pending' };
                                                return <span className={`aap-job-badge aap-job-badge--lg ${info.cls}`}>{info.label}</span>;
                                            })()}
                                            <span className="aap-job-detail-time">{fmtTime(activeJob.startedAt)}</span>
                                            {(activeJob.status === 'PENDING' || activeJob.status === 'STARTED') && (
                                                <span className="aap-live-badge">● Live</span>
                                            )}
                                        </div>

                                        {/* Task ID */}
                                        {activeJob.taskId && (
                                            <div className="aap-taskid">
                                                <span className="aap-taskid-label">Task ID</span>
                                                <span className="aap-taskid-val">{activeJob.taskId}</span>
                                            </div>
                                        )}

                                        {/* Progress bar */}
                                        <ProgressBar progress={activeJob.progress} />

                                        {/* Log */}
                                        <div className="aap-section-label" style={{ marginBottom: 6 }}>Log</div>
                                        <div className="aap-logs">
                                            {activeJob.logs.map((line, i) => (
                                                <div key={i} className="aap-log-line">{line}</div>
                                            ))}
                                            <div ref={logsEnd} />
                                        </div>

                                        {/* Result */}
                                        {activeJob.status === 'SUCCESS' && activeJob.result && (
                                            <div className="aap-result-card">
                                                <span className="aap-result-icon"><Check size={22} /></span>
                                                <div style={{ flex: 1 }}>
                                                    <p className="aap-result-title">
                                                        {activeJob.result.annotated_count} / {activeJob.result.total} images annotated
                                                    </p>
                                                    <div className="aap-result-breakdown">
                                                        {(activeJob.result.no_detection > 0) && (
                                                            <span className="aap-stat aap-stat--warn">
                                                                ⬜ {activeJob.result.no_detection} no detection
                                                            </span>
                                                        )}
                                                        {(activeJob.result.skipped_path > 0) && (
                                                            <span className="aap-stat aap-stat--err">
                                                                <AlertTriangle size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 2 }} /> {activeJob.result.skipped_path} file not found
                                                            </span>
                                                        )}
                                                    </div>
                                                    {activeJob.result.skipped_path > 0 && (
                                                        <p className="aap-result-hint">
                                                            Files not found means images are missing from disk. Check your upload folder.
                                                        </p>
                                                    )}
                                                    {activeJob.result.no_detection > 0 && activeJob.result.skipped_path === 0 && (
                                                        <p className="aap-result-hint">
                                                            Images with no detection need more training data. Annotate more examples and retrain the seed model.
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        {/* No worker */}
                                        {activeJob.status === 'NO_WORKER' && (
                                            <div className="aap-worker-err">
                                                <p>Worker not found. Start it then try again:</p>
                                                <div className="aap-cmd-block" style={{ marginTop: 8 }}>
                                                    <code>celery -A app.tasks.celery_app:celery_app worker --loglevel=info</code>
                                                    <button className="aap-cmd-copy" onClick={() => navigator.clipboard.writeText('celery -A app.tasks.celery_app:celery_app worker --loglevel=info')}><Clipboard size={14} /></button>
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
                <div className="aap-footer">
                    {view === 'setup' ? (
                        <button className="aap-start-btn" onClick={handleStart} disabled={!canStart}>
                            {launching
                                ? 'Starting…'
                                : <><Sparkles size={16} /> Auto-Annotate {selectedIds.size > 0 ? `${selectedIds.size} Image${selectedIds.size !== 1 ? 's' : ''}` : ''}</>}
                        </button>
                    ) : (
                        <button className="aap-start-btn aap-start-btn--secondary" onClick={() => { setView('setup'); loadSetup(); }}>
                            <ChevronLeft size={16} /> Back to Setup
                        </button>
                    )}
                    {anyRunning && (
                        <button className="aap-stop-btn" onClick={handleStop} title="Stop annotation">
                            <Square size={14} /> Stop
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AutoAnnotatePanel;
