import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Layers, X, RefreshCw, Eye, Play, Upload, FolderUp, Inbox } from 'lucide-react';
import { CLS_MODEL_GROUPS, DEFAULT_CLS_MODEL } from '../constants/yoloModels';
import AugmentationSettings, { useAug, augPayload } from './AugmentationSettings';
// Reuse MainTrainingPanel's styling (mtp-* classes) — same visual language.
import './MainTrainingPanel.css';

import { API_URL } from '../config';

const POLL_INTERVAL = 3000;
const IMPORT_CHUNK = 40;
const IMG_RE = /\.(jpe?g|png|bmp|webp)$/i;

const STAGE_LABEL = {
    reading: 'Reading annotations',
    dataset: 'Building classification set (cutting crops)',
    weights: 'Loading classifier weights (downloaded on first use)',
    classifier: 'Training classifier',
};

const STATUS_LABEL = {
    PENDING: { label: 'Pending', cls: 'badge--pending' },
    STARTED: { label: 'Running', cls: 'badge--running' },
    SUCCESS: { label: 'Done', cls: 'badge--done' },
    FAILURE: { label: 'Failed', cls: 'badge--fail' },
    REVOKED: { label: 'Stopped', cls: 'badge--fail' },
};
const DB_STATUS = { pending: 'PENDING', started: 'STARTED', success: 'SUCCESS', failure: 'FAILURE', revoked: 'REVOKED' };
const ACTIVE = ['PENDING', 'STARTED'];
const NO_WORKER_TICKS = 15; // ~45 s still PENDING => probably no worker running

const fmtTime = (d) => new Date(d).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

// Loss and validation accuracy per epoch (history comes from the shared epoch callback)
function ClsCharts({ history }) {
    if (!history?.length) return null;
    const hasLoss = history.some(h => h.loss != null || h['val/loss'] != null);
    const hasAcc = history.some(h => h.accuracy_top1 != null);
    const axis = { fill: '#666666', fontSize: 10 };
    return (
        <>
            {hasLoss && (
                <div className="chart-wrap">
                    <p className="chart-title">Loss</p>
                    <ResponsiveContainer width="100%" height={150}>
                        <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                            <XAxis dataKey="epoch" tick={axis} />
                            <YAxis tick={axis} />
                            <Tooltip />
                            <Legend wrapperStyle={{ fontSize: 10 }} />
                            <Line isAnimationActive={false} type="monotone" dataKey="loss" name="Train" stroke="#dc143c" dot={false} strokeWidth={1.5} />
                            <Line isAnimationActive={false} type="monotone" dataKey="val/loss" name="Validation" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
            {hasAcc && (
                <div className="chart-wrap">
                    <p className="chart-title">Validation accuracy</p>
                    <ResponsiveContainer width="100%" height={150}>
                        <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                            <XAxis dataKey="epoch" tick={axis} />
                            <YAxis domain={[0, 1]} tick={axis} />
                            <Tooltip />
                            <Legend wrapperStyle={{ fontSize: 10 }} />
                            <Line isAnimationActive={false} type="monotone" dataKey="accuracy_top1" name="Top-1" stroke="#10b981" dot={false} strokeWidth={1.5} />
                            <Line isAnimationActive={false} type="monotone" dataKey="accuracy_top5" name="Top-5" stroke="#38bdf8" dot={false} strokeWidth={1.5} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </>
    );
}

function LogBox({ lines }) {
    const ref = useRef(null);
    useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [lines]);
    if (!lines?.length) return null;
    return (
        <pre ref={ref} style={{ background: '#111', color: '#d4d4d4', fontSize: 11, lineHeight: 1.5, padding: 10,
                                borderRadius: 8, maxHeight: 220, overflow: 'auto', margin: '10px 0', whiteSpace: 'pre-wrap' }}>
            {lines.join('\n')}
        </pre>
    );
}

// Plain checkbox row (the mtp-toggle-* classes are styled for switch markup).
function CheckRow({ checked, onChange, disabled, children, style }) {
    return (
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, padding: '3px 0',
                        cursor: disabled ? 'default' : 'pointer', ...style }}>
            <input type="checkbox" checked={checked} disabled={disabled} onChange={onChange} />
            {children}
        </label>
    );
}

// Rows = true class, columns = predicted class.
function ConfusionTable({ matrix }) {
    const classes = Object.keys(matrix || {});
    if (!classes.length) return null;
    return (
        <table style={{ borderCollapse: 'collapse', fontSize: 13, marginTop: 8 }}>
            <thead>
                <tr>
                    <th style={{ padding: '4px 8px', textAlign: 'left' }}>true \ predicted</th>
                    {classes.map(c => <th key={c} style={{ padding: '4px 8px' }}>{c}</th>)}
                </tr>
            </thead>
            <tbody>
                {classes.map(t => (
                    <tr key={t}>
                        <td style={{ padding: '4px 8px', fontWeight: 600 }}>{t}</td>
                        {classes.map(p => (
                            <td key={p} style={{
                                padding: '4px 8px', textAlign: 'center',
                                background: t === p && matrix[t][p] ? 'rgba(34,197,94,0.2)'
                                    : matrix[t][p] ? 'rgba(239,68,68,0.2)' : 'transparent',
                            }}>{matrix[t][p]}</td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

export default function ClassifierTrainingPanel({ project, onClose }) {
    const [classCounts, setClassCounts] = useState({});
    const [loading, setLoading] = useState(true);
    const [modelStatus, setModelStatus] = useState(null);

    // Where the classes come from: boxes nested in a detected region, or one folder per class
    const [source, setSource] = useState('boxes');   // 'boxes' | 'folders'

    // Boxes mode: the trained detector finds the crop class; everything else inside becomes a class
    const [detector, setDetector] = useState('');    // 'main' | 'seed'
    const [cropClass, setCropClass] = useState('');
    const [labelClasses, setLabelClasses] = useState([]);
    const [emptyAsLabel, setEmptyAsLabel] = useState(true);
    const [emptyLabel, setEmptyLabel] = useState('no_cover');
    const [minOverlap, setMinOverlap] = useState(0.5);
    const [margin, setMargin] = useState(0.12);

    // Folders mode
    const [folderClasses, setFolderClasses] = useState([]);
    const [pending, setPending] = useState(null);    // {files, labels, perClass} chosen, not yet uploaded
    const [importing, setImporting] = useState(null); // {done, total}
    const [importResult, setImportResult] = useState(null);

    // Training config — same shape as the Main / Segmentation panels
    const [selectedModel, setSelectedModel] = useState(DEFAULT_CLS_MODEL);
    const [epochs, setEpochs] = useState(40);
    const [imgsz, setImgsz] = useState(0);      // 0 = Auto (sized from the crops)
    const [batch, setBatch] = useState(-1);     // -1 = Auto (fits the GPU)
    const [preprocess, setPreprocess] = useState(true);
    const [aug, setAug] = useAug();

    // Preview / job state
    const [preview, setPreview] = useState(null);
    const [previewing, setPreviewing] = useState(false);
    const [error, setError] = useState(null);
    const [view, setView] = useState('setup');      // 'setup' | 'jobs'
    const [jobs, setJobs] = useState([]);            // persisted classifier jobs, oldest first
    const [activeJobId, setActiveJobId] = useState(null);
    const jobsRef = useRef([]);
    jobsRef.current = jobs;

    // Test-image prediction
    const [prediction, setPrediction] = useState(null);
    const [predicting, setPredicting] = useState(false);

    const classNames = Object.keys(classCounts);
    const classKey = classNames.join('|');
    const detectors = modelStatus?.detectors || {};
    const hasDetector = !!(detectors.main || detectors.seed);

    const load = useCallback(() => {
        setLoading(true);
        Promise.all([
            axios.get(`${API_URL}/pipeline/training-stats/${project.id}`),
            axios.get(`${API_URL}/crop-cls/model-status/${project.id}`),
        ])
            .then(([stats, ms]) => {
                setClassCounts(stats.data.class_breakdown || {});
                setModelStatus(ms.data);
            })
            .catch(() => setError('Could not load project data.'))
            .finally(() => setLoading(false));
    }, [project.id]);

    useEffect(() => { load(); }, [load]);
    const patchJob = (id, patch) => setJobs(prev => prev.map(j => (j.id === id ? { ...j, ...patch } : j)));

    // Turn a Celery task-status reply into job state, and persist it when the job ends.
    const applyStatus = useCallback((job, data) => {
        const res = data.result;
        let status = data.status;
        let error = data.error || null;
        if (status === 'SUCCESS' && res?.error) { status = 'FAILURE'; error = res.error; }   // task returned {"error": ...}
        const meta = data.meta || null;
        const logs = res?.logs || meta?.logs || job.logs;
        const history = res?.history || meta?.history || job.history;
        const ticks = status === 'PENDING' ? (job.pendingTicks || 0) + 1 : 0;
        patchJob(job.id, { status, meta: meta || job.meta, result: status === 'SUCCESS' ? res : job.result,
                           error, logs, history, pendingTicks: ticks });
        if (!ACTIVE.includes(status)) {
            axios.patch(`${API_URL}/pipeline/jobs/${job.taskId}`, {
                status: status === 'SUCCESS' ? 'success' : status === 'REVOKED' ? 'revoked' : 'failure',
                result_meta: { logs, history, summary: job.summary, result: status === 'SUCCESS' ? res : undefined, error },
                finished_at: new Date().toISOString(),
            }).catch(() => {});
            if (status === 'SUCCESS') load();
        }
    }, [load]);   // eslint-disable-line react-hooks/exhaustive-deps

    // Saved jobs survive closing the panel / reloading the page
    useEffect(() => {
        axios.get(`${API_URL}/pipeline/jobs/${project.id}?job_type=cls_training`)
            .then(async res => {
                const loaded = res.data.map(j => ({
                    id: j.id, taskId: j.id,
                    status: DB_STATUS[j.status] || 'PENDING',
                    summary: j.result_meta?.summary || '',
                    logs: j.result_meta?.logs || [],
                    history: j.result_meta?.history || [],
                    result: j.result_meta?.result || null,
                    error: j.result_meta?.error || null,
                    meta: null,
                    startedAt: new Date(j.created_at),
                }));
                setJobs(loaded);
                if (loaded.length) setActiveJobId(loaded[loaded.length - 1].id);
                // anything that was running when the panel closed: ask Celery what happened
                for (const job of loaded.filter(j => ACTIVE.includes(j.status))) {
                    try {
                        const { data } = await axios.get(`${API_URL}/pipeline/task-status/${job.taskId}`);
                        applyStatus(job, data);
                    } catch { /* keep as is */ }
                }
            })
            .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [project.id]);

    // One poller for every running job
    useEffect(() => {
        const t = setInterval(async () => {
            for (const job of jobsRef.current.filter(j => ACTIVE.includes(j.status))) {
                try {
                    const { data } = await axios.get(`${API_URL}/pipeline/task-status/${job.taskId}`);
                    applyStatus(job, data);
                } catch { /* transient — keep polling */ }
            }
        }, POLL_INTERVAL);
        return () => clearInterval(t);
    }, [applyStatus]);

    // Default to the best trained detector (Main, else Seed).
    useEffect(() => {
        if (!detector && hasDetector) setDetector(detectors.main ? 'main' : 'seed');
    }, [detector, hasDetector, detectors.main]);

    // Default the crop class to the region-looking class once classes load.
    useEffect(() => {
        if (!cropClass && classNames.length) {
            setCropClass(classNames.find(c => /engine|region/i.test(c)) || classNames[0]);
        }
    }, [classKey, cropClass]);   // eslint-disable-line react-hooks/exhaustive-deps

    // Every class other than the crop class is a classifier class unless unticked.
    useEffect(() => {
        setLabelClasses(classNames.filter(c => c !== cropClass));
        setPreview(null);
    }, [classKey, cropClass]);   // eslint-disable-line react-hooks/exhaustive-deps

    const toggleLabel = (c) => {
        setPreview(null);
        setLabelClasses(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);
    };
    const toggleFolderClass = (c) =>
        setFolderClasses(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);

    const rules = () => ({
        crop_class: cropClass,
        label_classes: labelClasses,
        empty_label: emptyAsLabel && emptyLabel.trim() ? emptyLabel.trim() : null,
        min_overlap: minOverlap,
        margin,
    });

    const canRun = source === 'folders'
        ? folderClasses.length >= 2
        : !!(cropClass && labelClasses.length > 0);

    // ── Folder import ──
    const pickFolder = (fileList) => {
        const files = [], labels = [], perClass = {};
        Array.from(fileList || []).forEach(f => {
            if (!IMG_RE.test(f.name)) return;
            const parts = (f.webkitRelativePath || '').split('/').filter(Boolean);
            const label = parts.length >= 2 ? parts[parts.length - 2] : null;
            if (!label) return;
            files.push(f); labels.push(label);
            perClass[label] = (perClass[label] || 0) + 1;
        });
        setImportResult(null);
        setPending(files.length ? { files, labels, perClass } : null);
        setError(files.length ? null
            : 'No images found inside class folders. Pick a folder that contains one sub-folder per class.');
    };

    const importFolders = async (zipFile) => {
        setError(null);
        setImportResult(null);
        const jobs = zipFile
            ? [{ files: [zipFile], labels: null }]
            : Array.from({ length: Math.ceil(pending.files.length / IMPORT_CHUNK) }, (_, i) => ({
                files: pending.files.slice(i * IMPORT_CHUNK, (i + 1) * IMPORT_CHUNK),
                labels: pending.labels.slice(i * IMPORT_CHUNK, (i + 1) * IMPORT_CHUNK),
            }));
        const total = zipFile ? 1 : pending.files.length;
        const agg = { imported: 0, per_class: {}, failed: [] };
        let done = 0;
        setImporting({ done: 0, total });
        try {
            for (const j of jobs) {
                const form = new FormData();
                j.files.forEach(f => form.append('files', f));
                if (j.labels) form.append('labels', JSON.stringify(j.labels));
                const { data } = await axios.post(`${API_URL}/crop-cls/import-folders/${project.id}`, form);
                agg.imported += data.imported;
                agg.failed.push(...data.failed);
                Object.entries(data.per_class).forEach(([c, n]) => { agg.per_class[c] = (agg.per_class[c] || 0) + n; });
                done += j.files.length;
                setImporting({ done, total });
            }
            setImportResult(agg);
            setPending(null);
            setFolderClasses(prev => [...new Set([...prev, ...Object.keys(agg.per_class)])]);
            load();
        } catch (e) {
            setError(e.response?.data?.detail || 'Import failed.');
        } finally {
            setImporting(null);
        }
    };

    // ── Preview / train ──
    const runPreview = async () => {
        setError(null);
        setPreviewing(true);
        try {
            const res = await axios.post(`${API_URL}/crop-cls/preview/${project.id}`, rules());
            setPreview(res.data);
        } catch (e) {
            setPreview(null);
            setError(e.response?.data?.detail || 'Preview failed.');
        } finally {
            setPreviewing(false);
        }
    };

    const startTraining = async () => {
        setError(null);
        try {
            const common = {
                cls_model_name: selectedModel, cls_epochs: epochs, cls_imgsz: imgsz,
                batch, preprocess, ...augPayload(aug),
            };
            const body = source === 'folders'
                ? { ...common, mode: 'whole', region_classes: folderClasses }
                : { ...common, mode: 'crop', ...(detector ? { detector } : {}), ...rules() };
            const { data } = await axios.post(`${API_URL}/crop-cls/train/${project.id}`, body);
            const summary = source === 'folders'
                ? `${selectedModel} · image folders (${folderClasses.length} classes)`
                : `${selectedModel} · crop "${cropClass}" → ${labelClasses.length} classes`;
            const job = {
                id: data.task_id, taskId: data.task_id, status: 'PENDING', summary,
                logs: [`Task ID: ${data.task_id}`, 'Waiting for the worker to pick the job up…'],
                history: [], result: null, error: null, meta: null, startedAt: new Date(),
            };
            setJobs(prev => [...prev, job]);
            setActiveJobId(job.id);
            setView('jobs');
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: data.task_id, project_id: project.id, job_type: 'cls_training',
                result_meta: { logs: job.logs, summary, startedAt: job.startedAt.toISOString() },
            }).catch(() => {});
        } catch (e) {
            setError(e.response?.data?.detail || 'Could not start training.');
        }
    };

    const stopJob = async (job) => {
        await axios.post(`${API_URL}/crop-cls/stop/${job.taskId}`).catch(() => {});
    };

    const removeJob = (job) => {
        axios.delete(`${API_URL}/pipeline/jobs/${job.taskId}`).catch(() => {});
        setJobs(prev => prev.filter(j => j.id !== job.id));
        if (activeJobId === job.id) setActiveJobId(null);
    };

    const predictFile = async (file) => {
        if (!file) return;
        setPredicting(true);
        setPrediction(null);
        try {
            const form = new FormData();
            form.append('file', file);
            const { data } = await axios.post(`${API_URL}/crop-cls/predict/${project.id}`, form);
            setPrediction(data);
        } catch (e) {
            setPrediction({ status: 'error', detail: e.response?.data?.detail || 'Prediction failed.' });
        } finally {
            setPredicting(false);
        }
    };

    const anyRunning = jobs.some(j => ACTIVE.includes(j.status));
    const activeJob = jobs.find(j => j.id === activeJobId) || jobs[jobs.length - 1] || null;

    return (
        <div className="mtp-overlay" onClick={onClose}>
            <div className="mtp-panel" onClick={e => e.stopPropagation()}>
                <div className="mtp-header">
                    <div className="mtp-header-left">
                        <span className="mtp-header-icon"><Layers size={20} /></span>
                        <div>
                            <h2 className="mtp-title">Train Classifier</h2>
                            <p className="mtp-subtitle">{project.name}</p>
                        </div>
                    </div>
                    <button className="mtp-close" onClick={onClose}><X size={18} /></button>
                </div>

                <div className="mtp-tabs">
                    <button className={`mtp-tab ${view === 'setup' ? 'mtp-tab--active' : ''}`} onClick={() => setView('setup')}>
                        Setup
                    </button>
                    <button className={`mtp-tab ${view === 'jobs' ? 'mtp-tab--active' : ''}`} onClick={() => setView('jobs')}>
                        Jobs
                        {jobs.length > 0 && <span className="mtp-tab-badge">{jobs.length}</span>}
                        {anyRunning && <span className="mtp-tab-dot" />}
                    </button>
                </div>

                <div className="mtp-body">
                    {error && <div className="mtp-warning">{error}</div>}

                    {view === 'setup' && (<>
                    {/* ── 1. Data ── */}
                    <section className="mtp-section">
                        <div className="mtp-section-header">
                            <span className="mtp-section-title">1. Classification data</span>
                            <button className="mtp-refresh" onClick={load} title="Refresh"><RefreshCw size={15} /></button>
                        </div>
                        <div className="mtp-tabs" style={{ marginBottom: 12 }}>
                            <button className={`mtp-tab ${source === 'boxes' ? 'mtp-tab--active' : ''}`} onClick={() => setSource('boxes')}>
                                From annotations
                            </button>
                            <button className={`mtp-tab ${source === 'folders' ? 'mtp-tab--active' : ''}`} onClick={() => setSource('folders')}>
                                Image folders
                            </button>
                        </div>

                        {loading && <div className="mtp-loading"><div className="mtp-spinner" /><span>Loading…</span></div>}

                        {/* ── From annotations: trained detector → crop class → remaining classes ── */}
                        {!loading && source === 'boxes' && (
                            classNames.length === 0 ? (
                                <div className="mtp-warning">No annotated classes yet — annotate some images first.</div>
                            ) : (
                                <>
                                    <p style={{ fontSize: 13, opacity: 0.8, marginTop: 0 }}>
                                        Each box of the crop class you pick is cut out of the photo and labelled by the classes inside it; the classifier trains on those crops only. A trained detector is needed later, just to find the region when you test on a full image.
                                    </p>

                                    <div className="mtp-model-row">
                                        <label className="mtp-model-label">
                                            Detector for testing
                                            <span className="mtp-model-hint"> (not used for training)</span>
                                        </label>
                                        {hasDetector ? (
                                            <select className="mtp-model-select" value={detector} onChange={e => setDetector(e.target.value)}>
                                                {detectors.main && <option value="main">Main model</option>}
                                                {detectors.seed && <option value="seed">Seed model</option>}
                                            </select>
                                        ) : (
                                            <p style={{ fontSize: 12, opacity: 0.7, margin: 0 }}>
                                                No trained detector yet. You can still train; train a Seed or Main model that includes the crop class before testing on a full image.
                                            </p>
                                        )}
                                    </div>

                                    <div className="mtp-model-row">
                                        <label className="mtp-model-label">Crop class (detector class to cut out)</label>
                                        <select className="mtp-model-select" value={cropClass}
                                                onChange={e => setCropClass(e.target.value)}>
                                            {classNames.map(c => <option key={c} value={c}>{c} ({classCounts[c]})</option>)}
                                        </select>
                                    </div>

                                    <div style={{ margin: '12px 0 4px', fontWeight: 600 }}>
                                        Classifier classes (all remaining classes)
                                    </div>
                                    {classNames.filter(c => c !== cropClass).map(c => (
                                        <CheckRow key={c} checked={labelClasses.includes(c)} onChange={() => toggleLabel(c)}>
                                            {c} ({classCounts[c]})
                                        </CheckRow>
                                    ))}
                                    <CheckRow
                                        checked={emptyAsLabel}
                                        onChange={e => { setEmptyAsLabel(e.target.checked); setPreview(null); }}
                                        style={{ marginTop: 8 }}
                                    >
                                        Crop with nothing inside is labelled
                                        <input type="text" value={emptyLabel} disabled={!emptyAsLabel}
                                               onChange={e => { setEmptyLabel(e.target.value); setPreview(null); }}
                                               style={{ width: 120 }} />
                                    </CheckRow>
                                    {!emptyAsLabel && <div className="mtp-warning">Crops with nothing inside will be skipped.</div>}

                                    <div className="mtp-epochs-row" style={{ marginTop: 10 }}>
                                        <span>Min overlap: {Math.round(minOverlap * 100)}% of a class box inside the crop</span>
                                        <input type="range" min="0.2" max="1" step="0.05" value={minOverlap} className="mtp-epochs-slider"
                                               onChange={e => { setMinOverlap(+e.target.value); setPreview(null); }} />
                                    </div>
                                    <div className="mtp-epochs-row">
                                        <span>Crop margin: {Math.round(margin * 100)}%</span>
                                        <input type="range" min="0" max="0.4" step="0.02" value={margin} className="mtp-epochs-slider"
                                               onChange={e => { setMargin(+e.target.value); setPreview(null); }} />
                                    </div>

                                    <button className="mtp-train-btn" style={{ marginTop: 12 }}
                                            disabled={!canRun || previewing} onClick={runPreview}>
                                        <Eye size={16} /> {previewing ? 'Building preview…' : 'Preview training data'}
                                    </button>
                                </>
                            )
                        )}

                        {/* ── Image folders ── */}
                        {!loading && source === 'folders' && (
                            <>
                                <p style={{ fontSize: 13, opacity: 0.8, marginTop: 0 }}>
                                    One folder per class (e.g. <code>full_cover/</code>, <code>cut_cover/</code>, <code>no_cover/</code>).
                                    Each image is labelled by its folder and the whole image is classified.
                                </p>
                                <label className="mtp-train-btn" style={{ display: 'inline-flex', width: 'auto', cursor: 'pointer', marginRight: 8 }}>
                                    <FolderUp size={16} /> Choose folder
                                    <input type="file" hidden multiple webkitdirectory="" directory=""
                                           onChange={e => { pickFolder(e.target.files); e.target.value = ''; }} />
                                </label>
                                <label className="mtp-refresh" style={{ display: 'inline-flex', cursor: 'pointer', padding: '8px 12px' }}>
                                    or a .zip
                                    <input type="file" hidden accept=".zip"
                                           onChange={e => { if (e.target.files[0]) importFolders(e.target.files[0]); e.target.value = ''; }} />
                                </label>

                                {pending && (
                                    <div style={{ marginTop: 12 }}>
                                        <div style={{ fontWeight: 600, marginBottom: 6 }}>
                                            {pending.files.length} images in {Object.keys(pending.perClass).length} classes
                                        </div>
                                        <div className="mtp-stat-cards">
                                            {Object.entries(pending.perClass).map(([c, n]) => (
                                                <div key={c} className="mtp-stat-card mtp-stat-card--green">
                                                    <span className="mtp-stat-value">{n}</span>
                                                    <span className="mtp-stat-label">{c}</span>
                                                </div>
                                            ))}
                                        </div>
                                        <button className="mtp-train-btn" style={{ marginTop: 10 }}
                                                disabled={!!importing} onClick={() => importFolders(null)}>
                                            <Upload size={16} /> {importing ? `Uploading ${importing.done}/${importing.total}…` : 'Import into project'}
                                        </button>
                                    </div>
                                )}
                                {importing && !pending && <p>Uploading {importing.done}/{importing.total}…</p>}
                                {importResult && (
                                    <p style={{ fontSize: 13, marginTop: 10 }}>
                                        ✅ Imported {importResult.imported} images
                                        {importResult.failed.length > 0 && ` · ${importResult.failed.length} skipped`}
                                        {importResult.failed.slice(0, 3).map((f, i) => (
                                            <span key={i} style={{ display: 'block', opacity: 0.7 }}>{f.filename}: {f.reason}</span>
                                        ))}
                                    </p>
                                )}

                                <div style={{ margin: '14px 0 4px', fontWeight: 600 }}>Classes to train on</div>
                                {classNames.length === 0 && <p style={{ fontSize: 13, opacity: 0.7 }}>No classes yet — import a folder first.</p>}
                                {classNames.map(c => (
                                    <CheckRow key={c} checked={folderClasses.includes(c)} onChange={() => toggleFolderClass(c)}>
                                        {c} ({classCounts[c]})
                                    </CheckRow>
                                ))}
                                {folderClasses.length === 1 && <div className="mtp-warning">Pick at least 2 classes.</div>}
                            </>
                        )}
                    </section>

                    {/* ── Preview ── */}
                    {source === 'boxes' && preview && (
                        <section className="mtp-section">
                            <div className="mtp-section-header">
                                <span className="mtp-section-title">Training data preview</span>
                            </div>
                            {preview.warning && <div className="mtp-warning">{preview.warning}</div>}
                            <div className="mtp-stat-cards">
                                {Object.entries(preview.per_class).map(([c, n]) => (
                                    <div key={c} className="mtp-stat-card mtp-stat-card--green">
                                        <span className="mtp-stat-value">{n}</span>
                                        <span className="mtp-stat-label">{c}</span>
                                    </div>
                                ))}
                                <div className="mtp-stat-card mtp-stat-card--yellow">
                                    <span className="mtp-stat-value">{preview.conflict}</span>
                                    <span className="mtp-stat-label">Conflicts (skipped)</span>
                                </div>
                                {!emptyAsLabel && (
                                    <div className="mtp-stat-card mtp-stat-card--yellow">
                                        <span className="mtp-stat-value">{preview.empty}</span>
                                        <span className="mtp-stat-label">Empty (skipped)</span>
                                    </div>
                                )}
                            </div>
                            <p style={{ fontSize: 13, opacity: 0.8 }}>
                                {preview.images} of {preview.annotated_images} annotated images used
                                {preview.images_without_crop > 0 &&
                                    ` · ${preview.images_without_crop} have no “${cropClass}” box`}
                            </p>
                            {Object.entries(preview.samples || {}).map(([label, thumbs]) => (
                                <div key={label} style={{ marginTop: 8 }}>
                                    <div style={{ fontWeight: 600 }}>{label}</div>
                                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                                        {thumbs.map((src, i) => (
                                            <img key={i} src={src} alt={label} style={{ height: 90, borderRadius: 6 }} />
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </section>
                    )}

                    {/* ── 2. Training config (same shape as the Main / Segmentation panels) ── */}
                    <section className="mtp-section">
                        <div className="mtp-section-header">
                            <span className="mtp-section-title">2. Training config</span>
                        </div>

                        <div className="mtp-model-row">
                            <label className="mtp-model-label">Classifier model</label>
                            <select className="mtp-model-select" value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                                {CLS_MODEL_GROUPS.map(g => (
                                    <optgroup key={g.family} label={g.family}>
                                        {g.models.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                                    </optgroup>
                                ))}
                            </select>
                        </div>
                        <div className="mtp-epochs-row">
                            <span>Epochs: {epochs}</span>
                            <input type="range" min="10" max="200" step="5" value={epochs}
                                   className="mtp-epochs-slider" onChange={e => setEpochs(+e.target.value)} />
                        </div>
                        <div className="mtp-model-row">
                            <label className="mtp-model-label">
                                Image Size
                                <span className="mtp-model-hint"> (Auto sizes it from your crops)</span>
                            </label>
                            <select className="mtp-model-select mtp-model-select--sm" value={imgsz}
                                    onChange={e => setImgsz(Number(e.target.value))}>
                                <option value={0}>Auto (recommended)</option>
                                {[160, 224, 320, 384, 448, 640].map(n => <option key={n} value={n}>{n} × {n}</option>)}
                            </select>
                        </div>
                        <div className="mtp-model-row">
                            <label className="mtp-model-label">
                                Batch Size
                                <span className="mtp-model-hint"> (Auto finds max that fits in VRAM)</span>
                            </label>
                            <select className="mtp-model-select mtp-model-select--sm" value={batch}
                                    onChange={e => setBatch(Number(e.target.value))}>
                                <option value={-1}>Auto (recommended)</option>
                                {[2, 4, 8, 16, 32, 64].map(b => <option key={b} value={b}>{b}</option>)}
                            </select>
                        </div>
                        <CheckRow checked={preprocess} onChange={e => setPreprocess(e.target.checked)}>
                            Enhance images (CLAHE / gamma / sharpen) before training
                        </CheckRow>

                        <AugmentationSettings aug={aug} onChange={setAug} kind="cls" />

                        {modelStatus?.has_classifier && (
                            <p style={{ fontSize: 13, opacity: 0.8 }}>
                                ✅ Trained classifier exists
                                {modelStatus.meta?.val_accuracy != null &&
                                    ` — val accuracy ${(modelStatus.meta.val_accuracy * 100).toFixed(1)}%`}
                                {' '}(training again replaces it)
                            </p>
                        )}
                        {anyRunning && (
                            <p style={{ fontSize: 12, opacity: 0.75 }}>
                                A job is already running — a new one will wait for it (see the Jobs tab).
                            </p>
                        )}
                        <button className="mtp-train-btn" style={{ marginTop: 4 }}
                                disabled={!canRun} onClick={startTraining}>
                            <Play size={16} /> Train classifier
                        </button>
                    </section>

                    {/* ── Test ── */}
                    {modelStatus?.has_classifier && (
                        <section className="mtp-section">
                            <div className="mtp-section-header">
                                <span className="mtp-section-title">3. Test on an image</span>
                            </div>
                            <label className="mtp-train-btn" style={{ display: 'inline-flex', width: 'auto', cursor: 'pointer' }}>
                                <Upload size={16} /> {predicting ? 'Running…' : 'Choose image'}
                                <input type="file" accept="image/*" hidden
                                       onChange={e => predictFile(e.target.files[0])} />
                            </label>
                            {prediction && (
                                <div style={{ marginTop: 10 }}>
                                    {prediction.status === 'error' ? (
                                        <div className="mtp-warning">{prediction.detail}</div>
                                    ) : (
                                        <p>
                                            <b>{prediction.state || prediction.status}</b>
                                            {prediction.confidence != null &&
                                                ` (${(prediction.confidence * 100).toFixed(0)}%)`}
                                            {prediction.status === 'uncertain' && ' — low confidence'}
                                            {prediction.status === 'region_not_found' && ' — region not found'}
                                        </p>
                                    )}
                                    {prediction.preview &&
                                        <img src={prediction.preview} alt="prediction" style={{ maxWidth: '100%', borderRadius: 8 }} />}
                                </div>
                            )}
                        </section>
                    )}
                    </>)}

                    {/* ═══════════ JOBS VIEW ═══════════ */}
                    {view === 'jobs' && (
                        jobs.length === 0 ? (
                            <div className="mtp-jobs-empty">
                                <span className="mtp-jobs-empty-icon"><Inbox size={32} /></span>
                                <p>No classifier jobs yet.</p>
                                <p className="mtp-jobs-empty-sub">Click <strong>Train classifier</strong> in Setup to start one.</p>
                            </div>
                        ) : (
                            <div className="mtp-jobs-layout">
                                <div className="mtp-jobs-list">
                                    {[...jobs].reverse().map((job, idx) => {
                                        const info = STATUS_LABEL[job.status] || { label: job.status, cls: 'badge--pending' };
                                        return (
                                            <div key={job.id}
                                                 className={`mtp-job-item ${activeJob?.id === job.id ? 'mtp-job-item--active' : ''}`}
                                                 onClick={() => setActiveJobId(job.id)}>
                                                <div className="mtp-job-item-top">
                                                    <span className="mtp-job-num">#{jobs.length - idx}</span>
                                                    <span className={`mtp-job-badge ${info.cls}`}>{info.label}</span>
                                                    {!ACTIVE.includes(job.status) && (
                                                        <button onClick={(e) => { e.stopPropagation(); removeJob(job); }} title="Remove"
                                                                style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: '#999', padding: '0 2px', lineHeight: 1 }}>✕</button>
                                                    )}
                                                </div>
                                                <div className="mtp-job-item-model">{job.summary || 'Classifier'}</div>
                                                <div className="mtp-job-item-time">{fmtTime(job.startedAt)}</div>
                                                {job.status === 'STARTED' && job.meta?.epoch > 0 && (
                                                    <div className="mtp-job-item-epoch">{job.meta.epoch}/{job.meta.total_epochs}</div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>

                                {activeJob && (() => {
                                    const info = STATUS_LABEL[activeJob.status] || { label: activeJob.status, cls: 'badge--pending' };
                                    const meta = activeJob.meta || {};
                                    const dp = meta.dataset_progress;
                                    const hist = activeJob.history || [];
                                    const res = activeJob.status === 'SUCCESS' ? activeJob.result : null;
                                    const pct = meta.total_epochs ? Math.round(((meta.epoch || 0) / meta.total_epochs) * 100) : 0;
                                    return (
                                        <div className="mtp-job-detail">
                                            <div className="mtp-job-detail-header">
                                                <span className={`mtp-job-badge mtp-job-badge--lg ${info.cls}`}>{info.label}</span>
                                                <span className="mtp-job-detail-time">{fmtTime(activeJob.startedAt)}</span>
                                                {ACTIVE.includes(activeJob.status) && <span className="mtp-running-badge">● Live</span>}
                                                {ACTIVE.includes(activeJob.status) && (
                                                    <button className="mtp-refresh" style={{ marginLeft: 'auto' }} onClick={() => stopJob(activeJob)}>Stop</button>
                                                )}
                                            </div>
                                            {activeJob.summary && <div className="mtp-job-model-tag">{activeJob.summary}</div>}
                                            <div className="mtp-job-taskid">
                                                <span className="mtp-job-taskid-label">Task ID</span>
                                                <span className="mtp-job-taskid-val">{activeJob.taskId}</span>
                                            </div>

                                            {ACTIVE.includes(activeJob.status) && (
                                                <div style={{ margin: '10px 0' }}>
                                                    <div style={{ fontSize: 13, fontWeight: 600 }}>
                                                        {activeJob.status === 'PENDING' && !meta.stage
                                                            ? 'Waiting for the worker…'
                                                            : STAGE_LABEL[meta.stage] || 'Working…'}
                                                        {meta.stage === 'dataset' && dp ? ` — images ${dp.done}/${dp.total}` : ''}
                                                        {meta.stage === 'classifier' && meta.epoch > 0
                                                            ? ` — epoch ${meta.epoch}/${meta.total_epochs}`
                                                              + (meta.eta_seconds > 0 ? ` · ~${Math.ceil(meta.eta_seconds / 60)} min left` : '') : ''}
                                                    </div>
                                                    {meta.stage === 'classifier' && (
                                                        <div style={{ height: 6, background: '#eee', borderRadius: 4, marginTop: 6 }}>
                                                            <div style={{ width: `${pct}%`, height: '100%', background: '#dc143c', borderRadius: 4 }} />
                                                        </div>
                                                    )}
                                                    {activeJob.status === 'PENDING' && activeJob.pendingTicks >= NO_WORKER_TICKS && (
                                                        <div className="mtp-warning" style={{ marginTop: 8 }}>
                                                            The job has not started. Is the Celery worker running? Restart it after pulling new code,
                                                            and check its window for errors.
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {activeJob.error && <div className="mtp-warning">{activeJob.error}</div>}
                                            {activeJob.status === 'REVOKED' && <p>Stopped.</p>}

                                            <LogBox lines={activeJob.logs} />
                                            <ClsCharts history={hist} />

                                            {res && (
                                                <>
                                                    <p>
                                                        ✅ Done — validation accuracy{' '}
                                                        <b>{res.val_accuracy != null ? `${(res.val_accuracy * 100).toFixed(1)}%` : 'n/a'}</b>
                                                        {' '}on classes: {res.classes?.join(', ')}
                                                        {res.detector && ` · detector: ${res.detector} model`}
                                                    </p>
                                                    {res.detector_note && <div className="mtp-warning">{res.detector_note}</div>}
                                                    <ConfusionTable matrix={res.confusion} />
                                                </>
                                            )}
                                        </div>
                                    );
                                })()}
                            </div>
                        )
                    )}
                </div>
            </div>
        </div>
    );
}
