import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Layers, X, RefreshCw, Eye, Play, Upload, FolderUp } from 'lucide-react';
import AugmentToggles from './AugmentToggles';
// Reuse MainTrainingPanel's styling (mtp-* classes) — same visual language.
import './MainTrainingPanel.css';

import { API_URL } from '../config';

const POLL_INTERVAL = 3000;
const IMPORT_CHUNK = 40;
const IMG_RE = /\.(jpe?g|png|bmp|webp)$/i;

const STAGE_LABEL = {
    detector: 'Training region detector',
    dataset: 'Building classification set',
    classifier: 'Training classifier',
};

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

    // Labelling rules
    const [cropClass, setCropClass] = useState('');
    const [labelClasses, setLabelClasses] = useState([]);
    const [emptyAsLabel, setEmptyAsLabel] = useState(true);
    const [emptyLabel, setEmptyLabel] = useState('no_cover');
    const [minOverlap, setMinOverlap] = useState(0.5);
    const [margin, setMargin] = useState(0.12);

    // Where the classes come from: boxes nested in a region, or one folder per class
    const [source, setSource] = useState('boxes');   // 'boxes' | 'folders'
    const [folderClasses, setFolderClasses] = useState([]);
    const [pending, setPending] = useState(null);    // {files, labels, perClass} chosen, not yet uploaded
    const [importing, setImporting] = useState(null); // {done, total}
    const [importResult, setImportResult] = useState(null);

    // Augmentation
    const [augment, setAugment] = useState(true);
    const [rotate360, setRotate360] = useState(false);

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

    const canRunBoxes = cropClass && labelClasses.length > 0 && !labelClasses.includes(cropClass);
    const canRun = source === 'folders' ? folderClasses.length >= 2 : canRunBoxes;
    const toggleFolderClass = (c) =>
        setFolderClasses(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);

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
        if (!files.length) setError('No images found inside class folders. Pick a folder that contains one sub-folder per class.');
        else setError(null);
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
            const common = { cls_epochs: clsEpochs, det_epochs: detEpochs, augment, rotate_360: rotate360 };
            const body = source === 'folders'
                ? { ...common, mode: 'whole', region_classes: folderClasses }
                : { ...common, mode, ...rules() };
            const { data } = await axios.post(`${API_URL}/crop-cls/train/${project.id}`, body);
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
                        <div className="mtp-tabs" style={{ marginBottom: 12 }}>
                            <button className={`mtp-tab ${source === 'boxes' ? 'mtp-tab--active' : ''}`} onClick={() => setSource('boxes')}>
                                Boxes inside a region
                            </button>
                            <button className={`mtp-tab ${source === 'folders' ? 'mtp-tab--active' : ''}`} onClick={() => setSource('folders')}>
                                Image folders
                            </button>
                        </div>

                        {source === 'folders' && (
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
                                    <CheckRow key={c} checked={folderClasses.includes(c)} onChange={() => toggleFolderClass(c)}>{c} ({classCounts[c]})</CheckRow>
                                ))}
                                {folderClasses.length === 1 && <div className="mtp-warning">Pick at least 2 classes.</div>}
                            </>
                        )}

                        {source === 'boxes' && (loading ? (
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
                                    <CheckRow key={c} checked={labelClasses.includes(c)} onChange={() => toggleLabel(c)}>{c} ({classCounts[c]})</CheckRow>
                                ))}

                                <CheckRow
                                    checked={emptyAsLabel}
                                    onChange={e => { setEmptyAsLabel(e.target.checked); setPreview(null); }}
                                    style={{ marginTop: 10 }}
                                >
                                    Region with nothing inside is labelled
                                    <input
                                        type="text"
                                        value={emptyLabel}
                                        disabled={!emptyAsLabel}
                                        onChange={e => { setEmptyLabel(e.target.value); setPreview(null); }}
                                        style={{ width: 120 }}
                                    />
                                </CheckRow>
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
                        ))}
                    </section>

                    {/* ── Preview ── */}
                    {source === 'boxes' && preview && (
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
                        {source === 'boxes' && <div className="mtp-model-row">
                            <label className="mtp-model-label">Mode</label>
                            <select className="mtp-model-select" value={mode} onChange={e => setMode(e.target.value)}>
                                <option value="crop">Crop + classify (region detector, then classifier)</option>
                                <option value="whole">Whole image (baseline, no detector)</option>
                            </select>
                        </div>}
                        <AugmentToggles augment={augment} onAugment={setAugment}
                                        rotate360={rotate360} onRotate360={setRotate360} />
                        <div className="mtp-epochs-row">
                            <span>Classifier epochs: {clsEpochs}</span>
                            <input type="range" min="10" max="150" step="5" value={clsEpochs}
                                   className="mtp-epochs-slider" onChange={e => setClsEpochs(+e.target.value)} />
                        </div>
                        {source === 'boxes' && mode === 'crop' && (
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
                            style={{ marginTop: 12 }}
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
                </div>
            </div>
        </div>
    );
}
