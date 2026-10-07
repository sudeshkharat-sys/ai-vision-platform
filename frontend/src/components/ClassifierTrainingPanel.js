import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Layers, X, RefreshCw, Eye, Play, Upload } from 'lucide-react';
// Reuse MainTrainingPanel's styling (mtp-* classes) — same visual language.
import './MainTrainingPanel.css';

import { API_URL } from '../config';

const POLL_INTERVAL = 3000;

const STAGE_LABEL = {
    detector: 'Training region detector',
    dataset: 'Building classification set',
    classifier: 'Training classifier',
};

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

    // Labelling rules
    const [cropClass, setCropClass] = useState('');
    const [labelClasses, setLabelClasses] = useState([]);
    const [emptyAsLabel, setEmptyAsLabel] = useState(true);
    const [emptyLabel, setEmptyLabel] = useState('no_cover');
    const [minOverlap, setMinOverlap] = useState(0.5);
    const [margin, setMargin] = useState(0.12);

    // Training config
    const [mode, setMode] = useState('crop');
    const [clsEpochs, setClsEpochs] = useState(40);
    const [detEpochs, setDetEpochs] = useState(60);

    // Preview / job state
    const [preview, setPreview] = useState(null);
    const [previewing, setPreviewing] = useState(false);
    const [error, setError] = useState(null);
    const [job, setJob] = useState(null);   // {taskId, status, meta, result, error}
    const pollRef = useRef(null);

    // Test-image prediction
    const [prediction, setPrediction] = useState(null);
    const [predicting, setPredicting] = useState(false);

    const classNames = Object.keys(classCounts);

    const load = useCallback(() => {
        setLoading(true);
        Promise.all([
            axios.get(`${API_URL}/pipeline/training-stats/${project.id}`),
            axios.get(`${API_URL}/crop-cls/model-status/${project.id}`),
        ])
            .then(([stats, ms]) => {
                setClassCounts(stats.data.class_counts || {});
                setModelStatus(ms.data);
            })
            .catch(() => setError('Could not load project data.'))
            .finally(() => setLoading(false));
    }, [project.id]);

    useEffect(() => { load(); }, [load]);
    useEffect(() => () => clearInterval(pollRef.current), []);

    // Default the crop class to the biggest-looking name once classes load.
    useEffect(() => {
        if (!cropClass && classNames.length) {
            const guess = classNames.find(c => /engine|region/i.test(c)) || classNames[0];
            setCropClass(guess);
        }
    }, [classNames, cropClass]);

    const toggleLabel = (c) => {
        setPreview(null);
        setLabelClasses(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);
    };

    const rules = () => ({
        crop_class: cropClass,
        label_classes: labelClasses,
        empty_label: emptyAsLabel && emptyLabel.trim() ? emptyLabel.trim() : null,
        min_overlap: minOverlap,
        margin,
    });

    const canRun = cropClass && labelClasses.length > 0 && !labelClasses.includes(cropClass);

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

    const poll = (taskId) => {
        clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
            try {
                const { data } = await axios.get(`${API_URL}/pipeline/task-status/${taskId}`);
                setJob({ taskId, status: data.status, meta: data.meta, result: data.result, error: data.error });
                if (['SUCCESS', 'FAILURE', 'REVOKED'].includes(data.status)) {
                    clearInterval(pollRef.current);
                    if (data.status === 'SUCCESS') load();
                }
            } catch { /* transient — keep polling */ }
        }, POLL_INTERVAL);
    };

    const startTraining = async () => {
        setError(null);
        try {
            const { data } = await axios.post(`${API_URL}/crop-cls/train/${project.id}`, {
                mode,
                cls_epochs: clsEpochs,
                det_epochs: detEpochs,
                ...rules(),
            });
            setJob({ taskId: data.task_id, status: 'PENDING' });
            poll(data.task_id);
        } catch (e) {
            setError(e.response?.data?.detail || 'Could not start training.');
        }
    };

    const stopTraining = async () => {
        if (!job?.taskId) return;
        await axios.post(`${API_URL}/crop-cls/stop/${job.taskId}`).catch(() => {});
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

    const running = job && ['PENDING', 'STARTED'].includes(job.status);
    const result = job?.status === 'SUCCESS' ? job.result : null;
    const resultError = result?.error || (job?.status === 'FAILURE' ? job.error : null);

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

                <div className="mtp-body">
                    {error && <div className="mtp-warning">{error}</div>}

                    {/* ── 1. Classification set ── */}
                    <section className="mtp-section">
                        <div className="mtp-section-header">
                            <span className="mtp-section-title">1. Build the classification set</span>
                            <button className="mtp-refresh" onClick={load} title="Refresh"><RefreshCw size={15} /></button>
                        </div>
                        {loading ? (
                            <div className="mtp-loading"><div className="mtp-spinner" /><span>Loading…</span></div>
                        ) : classNames.length === 0 ? (
                            <div className="mtp-warning">No annotated classes yet — annotate some images first.</div>
                        ) : (
                            <>
                                <div className="mtp-model-row">
                                    <label className="mtp-model-label">Crop class (the region to cut out)</label>
                                    <select
                                        className="mtp-model-select"
                                        value={cropClass}
                                        onChange={e => {
                                            setCropClass(e.target.value);
                                            setLabelClasses(prev => prev.filter(c => c !== e.target.value));
                                            setPreview(null);
                                        }}
                                    >
                                        {classNames.map(c => (
                                            <option key={c} value={c}>{c} ({classCounts[c]})</option>
                                        ))}
                                    </select>
                                </div>

                                <div style={{ margin: '12px 0 4px', fontWeight: 600 }}>
                                    Label classes (boxes inside the region that name its state)
                                </div>
                                {classNames.filter(c => c !== cropClass).map(c => (
                                    <label key={c} className="mtp-toggle-row">
                                        <span className="mtp-toggle-label">
                                            <input
                                                type="checkbox"
                                                checked={labelClasses.includes(c)}
                                                onChange={() => toggleLabel(c)}
                                            />
                                            <span className="mtp-toggle-text">{c} ({classCounts[c]})</span>
                                        </span>
                                    </label>
                                ))}

                                <label className="mtp-toggle-row" style={{ marginTop: 10 }}>
                                    <span className="mtp-toggle-label">
                                        <input
                                            type="checkbox"
                                            checked={emptyAsLabel}
                                            onChange={e => { setEmptyAsLabel(e.target.checked); setPreview(null); }}
                                        />
                                        <span className="mtp-toggle-text">
                                            Region with nothing inside is labelled
                                        </span>
                                    </span>
                                    <input
                                        type="text"
                                        value={emptyLabel}
                                        disabled={!emptyAsLabel}
                                        onChange={e => { setEmptyLabel(e.target.value); setPreview(null); }}
                                        style={{ marginLeft: 8, width: 120 }}
                                    />
                                </label>
                                {!emptyAsLabel && (
                                    <div className="mtp-warning">Empty regions will be skipped.</div>
                                )}

                                <div className="mtp-epochs-row" style={{ marginTop: 10 }}>
                                    <span>Min overlap: {Math.round(minOverlap * 100)}% of a label box inside the region</span>
                                    <input
                                        type="range" min="0.2" max="1" step="0.05"
                                        value={minOverlap} className="mtp-epochs-slider"
                                        onChange={e => { setMinOverlap(+e.target.value); setPreview(null); }}
                                    />
                                </div>
                                <div className="mtp-epochs-row">
                                    <span>Crop margin: {Math.round(margin * 100)}%</span>
                                    <input
                                        type="range" min="0" max="0.4" step="0.02"
                                        value={margin} className="mtp-epochs-slider"
                                        onChange={e => { setMargin(+e.target.value); setPreview(null); }}
                                    />
                                </div>

                                <button
                                    className="mtp-train-btn"
                                    style={{ marginTop: 12 }}
                                    disabled={!canRun || previewing}
                                    onClick={runPreview}
                                >
                                    <Eye size={16} /> {previewing ? 'Building preview…' : 'Preview set'}
                                </button>
                            </>
                        )}
                    </section>

                    {/* ── Preview ── */}
                    {preview && (
                        <section className="mtp-section">
                            <div className="mtp-section-header">
                                <span className="mtp-section-title">Preview</span>
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
                                            <img key={i} src={src} alt={label}
                                                 style={{ height: 90, borderRadius: 6 }} />
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </section>
                    )}

                    {/* ── 2. Train ── */}
                    <section className="mtp-section">
                        <div className="mtp-section-header">
                            <span className="mtp-section-title">2. Train</span>
                        </div>
                        <div className="mtp-model-row">
                            <label className="mtp-model-label">Mode</label>
                            <select className="mtp-model-select" value={mode} onChange={e => setMode(e.target.value)}>
                                <option value="crop">Crop + classify (region detector, then classifier)</option>
                                <option value="whole">Whole image (baseline, no detector)</option>
                            </select>
                        </div>
                        <div className="mtp-epochs-row">
                            <span>Classifier epochs: {clsEpochs}</span>
                            <input type="range" min="10" max="150" step="5" value={clsEpochs}
                                   className="mtp-epochs-slider" onChange={e => setClsEpochs(+e.target.value)} />
                        </div>
                        {mode === 'crop' && (
                            <div className="mtp-epochs-row">
                                <span>Region detector epochs: {detEpochs}</span>
                                <input type="range" min="10" max="200" step="5" value={detEpochs}
                                       className="mtp-epochs-slider" onChange={e => setDetEpochs(+e.target.value)} />
                            </div>
                        )}
                        {modelStatus?.has_classifier && (
                            <p style={{ fontSize: 13, opacity: 0.8 }}>
                                ✅ Trained model exists
                                {modelStatus.meta?.val_accuracy != null &&
                                    ` — val accuracy ${(modelStatus.meta.val_accuracy * 100).toFixed(1)}%`}
                                {' '}(training again replaces it)
                            </p>
                        )}
                        <button
                            className="mtp-train-btn"
                            disabled={!canRun || running}
                            onClick={startTraining}
                        >
                            <Play size={16} /> {running ? 'Training…' : 'Train classifier'}
                        </button>
                        {running && (
                            <button className="mtp-refresh" style={{ marginLeft: 8 }} onClick={stopTraining}>
                                Stop
                            </button>
                        )}
                    </section>

                    {/* ── Progress / result ── */}
                    {job && (
                        <section className="mtp-section">
                            <div className="mtp-section-header">
                                <span className="mtp-section-title">Progress</span>
                            </div>
                            {running && (
                                <p>
                                    {STAGE_LABEL[job.meta?.stage] || 'Waiting for worker…'}
                                    {job.meta?.total_epochs
                                        ? ` — epoch ${job.meta.epoch || 0}/${job.meta.total_epochs}` : ''}
                                </p>
                            )}
                            {resultError && <div className="mtp-warning">{resultError}</div>}
                            {job.status === 'REVOKED' && <p>Stopped.</p>}
                            {result && !result.error && (
                                <>
                                    <p>
                                        ✅ Done — validation accuracy{' '}
                                        <b>{result.val_accuracy != null
                                            ? `${(result.val_accuracy * 100).toFixed(1)}%` : 'n/a'}</b>
                                        {' '}on classes: {result.classes?.join(', ')}
                                    </p>
                                    <ConfusionTable matrix={result.confusion} />
                                </>
                            )}
                        </section>
                    )}

                    {/* ── Test ── */}
                    {modelStatus?.has_classifier && (
                        <section className="mtp-section">
                            <div className="mtp-section-header">
                                <span className="mtp-section-title">3. Test on an image</span>
                            </div>
                            <label className="mtp-train-btn" style={{ display: 'inline-flex', cursor: 'pointer' }}>
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
                </div>
            </div>
        </div>
    );
}
