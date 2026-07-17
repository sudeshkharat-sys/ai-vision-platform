import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Type, X, Download, Play, Square, RefreshCw, Check, AlertTriangle } from 'lucide-react';
import './OcrTrainingPanel.css';

import { API_URL } from '../config';

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

/* Simple accuracy sparkline over epochs */
function AccSparkline({ history }) {
    const vals = (history || []).map(h => h.val_accuracy).filter(v => v != null);
    if (vals.length < 2) return null;
    const W = 260, H = 56;
    const min = Math.min(...vals), max = Math.max(...vals);
    const range = max - min || 0.01;
    const pts = vals.map((v, i) =>
        `${((i / (vals.length - 1)) * W).toFixed(1)},${(H - ((v - min) / range) * (H - 6) - 3).toFixed(1)}`
    ).join(' ');
    return (
        <svg width={W} height={H} className="ocr-sparkline">
            <polyline points={pts} fill="none" stroke="#4ade80" strokeWidth="2"
                strokeLinejoin="round" strokeLinecap="round" />
        </svg>
    );
}

const OcrTrainingPanel = ({ project, onClose }) => {
    const [stats, setStats]       = useState(null);
    const [modelInfo, setModelInfo] = useState(null);
    const [error, setError]       = useState(null);

    // engine: 'cnn' (TensorFlow character classifier) or 'tesseract' (fine-tuned traineddata)
    const [engine, setEngine]     = useState('cnn');

    // training settings
    const [epochs, setEpochs]     = useState(50);
    const [maxIterations, setMaxIterations] = useState(800);
    const [target, setTarget]     = useState(300);
    const [imgSize, setImgSize]   = useState(64);
    const [fineTune, setFineTune] = useState(false);
    const [usePretrained, setUsePretrained] = useState(true);
    const [focusChars, setFocusChars] = useState([]);

    // running job
    const [taskId, setTaskId]     = useState(null);
    const [testing, setTesting]   = useState(false);
    const [testResult, setTestResult] = useState(null);
    const [running, setRunning]   = useState(false);
    const [meta, setMeta]         = useState(null);   // live progress meta
    const [result, setResult]     = useState(null);   // final task result
    const pollRef = useRef(null);

    const loadStats = useCallback(() => {
        axios.get(`${API_URL}/ocr/dataset-stats/${project.id}`)
            .then(res => setStats(res.data))
            .catch(() => setError('Could not load character dataset stats.'));
        axios.get(`${API_URL}/ocr/model-status/${project.id}`)
            .then(res => setModelInfo(res.data))
            .catch(() => {});
    }, [project.id]);

    useEffect(() => { loadStats(); }, [loadStats]);
    useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

    const pollTask = useCallback((id) => {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/pipeline/task-status/${id}`);
                const { status, meta: m, result: r, error: err } = res.data;
                if (status === 'STARTED' && m) setMeta(m);
                if (status === 'SUCCESS') {
                    clearInterval(pollRef.current);
                    setRunning(false);
                    if (r?.error) { setError(r.error); }
                    else {
                        setResult(r);
                        loadStats();
                    }
                    axios.patch(`${API_URL}/pipeline/jobs/${id}`, {
                        status: r?.error ? 'failure' : 'success',
                        result_meta: r || {},
                        finished_at: new Date().toISOString(),
                    }).catch(() => {});
                } else if (status === 'FAILURE' || status === 'REVOKED') {
                    clearInterval(pollRef.current);
                    setRunning(false);
                    setError(err || 'Training failed or was stopped.');
                    axios.patch(`${API_URL}/pipeline/jobs/${id}`, {
                        status: 'failure', finished_at: new Date().toISOString(),
                    }).catch(() => {});
                }
            } catch { /* transient — keep polling */ }
        }, 2000);
    }, [loadStats]);

    const startTraining = async () => {
        setError(null); setResult(null); setMeta(null);
        try {
            const res = engine === 'tesseract'
                ? await axios.post(`${API_URL}/ocr/train-tesseract/${project.id}`, {
                    max_iterations: maxIterations,
                })
                : await axios.post(`${API_URL}/ocr/train/${project.id}`, {
                    epochs: fineTune ? Math.min(epochs, 25) : epochs,
                    target_per_class: target,
                    img_size: imgSize,
                    fine_tune: fineTune,
                    focus_classes: fineTune ? focusChars : [],
                    use_pretrained: usePretrained,
                });
            setTaskId(res.data.task_id);
            setRunning(true);
            axios.post(`${API_URL}/pipeline/jobs`, {
                task_id: res.data.task_id,
                project_id: project.id,
                job_type: engine === 'tesseract' ? 'ocr_tesseract_training' : 'ocr_training',
            }).catch(() => {});
            pollTask(res.data.task_id);
        } catch (e) {
            setError(e.response?.data?.detail || 'Could not start OCR training.');
        }
    };

    const stopTraining = () => {
        if (!taskId) return;
        axios.post(`${API_URL}/pipeline/cancel/${taskId}`).catch(() => {});
    };

    const handleTestImage = async (e) => {
        const f = e.target.files?.[0];
        e.target.value = ''; // allow re-selecting the same file
        if (!f) return;
        setTesting(true); setTestResult(null); setError(null);
        try {
            const form = new FormData();
            form.append('file', f);
            const res = await axios.post(
                `${API_URL}/ocr/predict/${project.id}?engine=${engine}`, form, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setTestResult(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Prediction failed.');
        } finally {
            setTesting(false);
        }
    };

    const download = async (fileType, filename) => {
        try {
            const res = await axios.get(
                `${API_URL}/ocr/download/${project.id}/${fileType}`,
                { responseType: 'blob' }
            );
            const url = URL.createObjectURL(new Blob([res.data]));
            const link = document.createElement('a');
            link.href = url; link.download = filename;
            document.body.appendChild(link); link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch {
            setError(`Failed to download ${filename}.`);
        }
    };

    const charCounts = stats?.char_counts || {};
    const chars = Object.keys(charCounts);
    const maxCount = Math.max(1, ...Object.values(charCounts));
    const history = meta?.history || result?.history || [];
    const finalMeta = result || modelInfo?.meta;

    return (
        <div className="ocr-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
            <div className="ocr-panel">
                <div className="ocr-header">
                    <div className="ocr-header-left">
                        <div className="ocr-header-icon"><Type size={20} /></div>
                        <div>
                            <h2>OCR Character Training</h2>
                            <p>{project.name} — train a character reader (.tflite) from your labeled boxes</p>
                        </div>
                    </div>
                    <div className="ocr-header-actions">
                        <button className="ocr-btn-icon" onClick={loadStats} title="Refresh"><RefreshCw size={15} /></button>
                        <button className="ocr-btn-icon" onClick={onClose}><X size={18} /></button>
                    </div>
                </div>

                <div className="ocr-body">
                    {error && (
                        <div className="ocr-error">
                            <AlertTriangle size={15} /> {error}
                            <button onClick={() => setError(null)}><X size={14} /></button>
                        </div>
                    )}

                    {/* ── Dataset overview ──────────────────────── */}
                    <div className="ocr-section">
                        <h3>Character dataset</h3>
                        {chars.length === 0 ? (
                            <p className="ocr-hint">
                                No single-character labels found yet. In the annotation workspace,
                                zoom into a plate photo, draw one box around each character and
                                label it with that character (e.g. <b>U</b>, <b>V</b>, <b>4</b>).
                                Each box becomes one training image automatically.
                            </p>
                        ) : (
                            <>
                                <p className="ocr-hint">
                                    {stats.total_chars} labeled characters, {chars.length} classes,
                                    from {stats.annotated_images} annotated photos.
                                    Under-represented characters are balanced by augmentation
                                    (rotation, lighting, blur, stroke thickness) up to {target}/class.
                                </p>
                                <div className="ocr-bars">
                                    {chars.map(c => (
                                        <div key={c} className="ocr-bar-col" title={`${c}: ${charCounts[c]}`}>
                                            <div className="ocr-bar-track">
                                                <div className="ocr-bar-fill"
                                                    style={{ height: `${(charCounts[c] / maxCount) * 100}%` }} />
                                            </div>
                                            <span className="ocr-bar-char">{c}</span>
                                            <span className="ocr-bar-count">{charCounts[c]}</span>
                                        </div>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>

                    {/* ── Engine choice ─────────────────────────── */}
                    <div className="ocr-section">
                        <h3>OCR engine</h3>
                        <div className="ocr-engine-choice">
                            <label className={`ocr-engine-card ${engine === 'cnn' ? 'selected' : ''}`}>
                                <input type="radio" name="ocr-engine" value="cnn"
                                    checked={engine === 'cnn'} disabled={running}
                                    onChange={() => { setEngine('cnn'); setResult(null); setTestResult(null); }} />
                                <div>
                                    <b>TensorFlow character model (.tflite)</b>
                                    <p>Reads one character at a time from your labeled boxes.
                                       Pretrained on EMNIST + synthetic engraved characters.
                                       Runs on-device with tflite_flutter.</p>
                                </div>
                            </label>
                            <label className={`ocr-engine-card ${engine === 'tesseract' ? 'selected' : ''}`}>
                                <input type="radio" name="ocr-engine" value="tesseract"
                                    checked={engine === 'tesseract'} disabled={running}
                                    onChange={() => { setEngine('tesseract'); setResult(null); setTestResult(null); }} />
                                <div>
                                    <b>Tesseract fine-tune (.traineddata)</b>
                                    <p>Starts from Google's pretrained eng model (tessdata_best) and
                                       fine-tunes it on your labeled boxes, grouped into text lines
                                       (the tesstrain flow). Drop-in file for flutter_tesseract_ocr.</p>
                                </div>
                            </label>
                        </div>
                    </div>

                    {/* ── Settings + start ──────────────────────── */}
                    <div className="ocr-section">
                        <h3>Training</h3>
                        {engine === 'tesseract' ? (
                        <div className="ocr-settings">
                            <label>Max iterations
                                <input type="number" min="100" max="10000" step="100"
                                    value={maxIterations} disabled={running}
                                    onChange={e => setMaxIterations(+e.target.value || 800)} />
                            </label>
                        </div>
                        ) : (<>
                        <div className="ocr-settings">
                            <label>Epochs
                                <input type="number" min="5" max="300" value={epochs}
                                    disabled={running}
                                    onChange={e => setEpochs(+e.target.value || 50)} />
                            </label>
                            <label>Samples per class (after augmentation)
                                <input type="number" min="50" max="5000" step="50" value={target}
                                    disabled={running}
                                    onChange={e => setTarget(+e.target.value || 300)} />
                            </label>
                            <label>Character image size
                                <select value={imgSize} disabled={running}
                                    onChange={e => setImgSize(+e.target.value)}>
                                    <option value={48}>48 × 48</option>
                                    <option value={64}>64 × 64 (recommended)</option>
                                    <option value={96}>96 × 96</option>
                                </select>
                            </label>
                        </div>

                        {!fineTune && (
                            <div className="ocr-finetune">
                                <label className="ocr-finetune-toggle">
                                    <input type="checkbox" checked={usePretrained} disabled={running}
                                        onChange={e => setUsePretrained(e.target.checked)} />
                                    <span>
                                        <b>Head start with character knowledge (recommended)</b> —
                                        the model first learns 0–9/A–Z from the EMNIST dataset (~700k real
                                        labeled character images from NIST) plus this platform's own
                                        computer-drawn engraved-style characters; your labeled photos then
                                        teach it your exact font and metal. First use downloads EMNIST
                                        (~0.5 GB) and builds the base once, then it's reused instantly.
                                    </span>
                                </label>
                            </div>
                        )}

                        {modelInfo?.has_model && (
                            <div className="ocr-finetune">
                                <label className="ocr-finetune-toggle">
                                    <input type="checkbox" checked={fineTune} disabled={running}
                                        onChange={e => {
                                            setFineTune(e.target.checked);
                                            if (e.target.checked && focusChars.length === 0 && modelInfo?.meta?.per_class_accuracy) {
                                                // Pre-select characters below 95% from the last run
                                                setFocusChars(Object.entries(modelInfo.meta.per_class_accuracy)
                                                    .filter(([, a]) => a < 0.95).map(([c]) => c));
                                            }
                                        }} />
                                    <span>
                                        <b>Fine-tune existing model</b> — continue from the trained model
                                        (faster, keeps its knowledge) and push harder on weak characters
                                    </span>
                                </label>
                                {fineTune && (
                                    <>
                                        <p className="ocr-hint">
                                            Focus characters (click to toggle) — these get ~2.5× more
                                            training samples. The rest still train normally, so nothing is forgotten:
                                        </p>
                                        <div className="ocr-focus-chips">
                                            {(modelInfo?.meta?.classes || chars).map(c => {
                                                const acc = modelInfo?.meta?.per_class_accuracy?.[c];
                                                const on = focusChars.includes(c);
                                                return (
                                                    <button key={c}
                                                        className={`ocr-chip ocr-chip--toggle ${on ? 'selected' : ''}`}
                                                        disabled={running}
                                                        title={acc != null ? `last accuracy ${pct(acc)}` : ''}
                                                        onClick={() => setFocusChars(prev =>
                                                            on ? prev.filter(x => x !== c) : [...prev, c])}>
                                                        {c}{acc != null && acc < 0.95 ? '!' : ''}
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </>
                                )}
                            </div>
                        )}
                        </>)}

                        {!running ? (
                            <button className="ocr-btn-train" onClick={startTraining}
                                disabled={chars.length < 2}>
                                <Play size={15} /> {engine === 'tesseract'
                                    ? 'Fine-tune Tesseract (eng → engraved)'
                                    : fineTune ? 'Fine-tune model' : 'Train character model'}
                            </button>
                        ) : (
                            <div className="ocr-progress">
                                <div className="ocr-progress-row">
                                    <span className="ocr-spinner" />
                                    <span>
                                        {meta?.phase === 'extracting_lines' &&
                                            `Building text lines… ${meta.current}/${meta.total} photos (${meta.lines ?? 0} lines)`}
                                        {meta?.phase === 'preparing' && (meta.detail || 'Preparing…')}
                                        {(meta?.phase === 'preparing_train_lines' || meta?.phase === 'preparing_eval_lines') &&
                                            `Preparing Tesseract training files… ${meta.current}/${meta.total}`}
                                        {meta?.phase === 'evaluating' && 'Evaluating on held-out lines…'}
                                        {meta?.phase === 'training' && meta?.engine === 'tesseract' &&
                                            `Fine-tuning Tesseract — iteration ${meta.iteration ?? 0}/${meta.total_iterations}` +
                                            (meta.char_error != null ? ` (char error ${meta.char_error}%)` : '')}
                                        {meta?.phase === 'extracting_characters' &&
                                            `Cutting characters… ${meta.current}/${meta.total} photos`}
                                        {meta?.phase === 'augmenting' && 'Balancing classes with augmentation…'}
                                        {meta?.phase === 'pretraining' &&
                                            `Building pre-trained base (one-time): ${meta.detail || ''}` +
                                            (meta.epoch ? ` — epoch ${meta.epoch}/${meta.total_epochs}` :
                                             meta.current ? ` ${meta.current}/${meta.total}` : '')}
                                        {(!meta || (meta?.phase === 'training' && meta?.engine !== 'tesseract')) &&
                                            `Training — epoch ${meta?.epoch ?? 0}/${meta?.total_epochs ?? epochs}` +
                                            (meta?.batch && meta?.total_batches
                                                ? `, batch ${meta.batch}/${meta.total_batches}` : '') +
                                            (meta?.eta_seconds ? ` (~${Math.ceil(meta.eta_seconds / 60)} min left)` : '')}
                                    </span>
                                    <button className="ocr-btn-stop" onClick={stopTraining}>
                                        <Square size={13} /> Stop
                                    </button>
                                </div>
                                {history.length > 0 && (
                                    <div className="ocr-live">
                                        <AccSparkline history={history} />
                                        <span>val accuracy: {pct(history[history.length - 1]?.val_accuracy)}</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* ── Results / trained model ───────────────── */}
                    {engine === 'cnn' && (result || modelInfo?.has_model) && (
                        <div className="ocr-section">
                            <h3>
                                {result ? <><Check size={15} className="ocr-ok" /> Training complete</> : 'Trained model'}
                            </h3>
                            {finalMeta && (
                                <div className="ocr-results">
                                    <div className="ocr-result-big">
                                        <span>{pct(finalMeta.val_accuracy)}</span>
                                        <small>validation accuracy</small>
                                    </div>
                                    {finalMeta.per_class_accuracy && (
                                        <div className="ocr-perclass">
                                            {Object.entries(finalMeta.per_class_accuracy).map(([c, a]) => (
                                                <span key={c}
                                                    className={`ocr-chip ${a >= 0.95 ? 'ok' : a >= 0.8 ? 'warn' : 'bad'}`}>
                                                    {c} {pct(a)}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                    {result?.started_from === 'pretrained_base' && (
                                        <p className="ocr-hint">
                                            {result?.base_info?.emnist_used
                                                ? `Base knowledge: EMNIST (${result.base_info.emnist_samples} real characters) + ${result.base_info.synthetic_samples} synthetic.`
                                                : 'Base knowledge: synthetic characters only — EMNIST was NOT downloaded (check server internet).'}
                                        </p>
                                    )}
                                    {result?.top_confusions?.length > 0 && (
                                        <p className="ocr-hint">
                                            Most confused: {result.top_confusions.slice(0, 5)
                                                .map(t => `${t.pair.replace('->', ' → ')} (${t.count})`).join(', ')}.
                                            Label more examples of these characters and retrain.
                                        </p>
                                    )}
                                </div>
                            )}
                            <div className="ocr-downloads">
                                <button onClick={() => download('tflite', 'ocr_model.tflite')}>
                                    <Download size={14} /> ocr_model.tflite
                                </button>
                                <button onClick={() => download('labels', 'labels.txt')}>
                                    <Download size={14} /> labels.txt
                                </button>
                                <button onClick={() => download('meta', 'ocr_meta.json')}>
                                    <Download size={14} /> ocr_meta.json
                                </button>
                            </div>
                            {/* ── Test window ─────────────────── */}
                            <div className="ocr-test">
                                <h3>Test the model</h3>
                                <p className="ocr-hint">
                                    Upload a plate photo — the characters are found and read with your
                                    trained model. For an honest test, use a photo the model was
                                    NOT trained on.
                                </p>
                                <label className={`ocr-btn-test ${testing ? 'busy' : ''}`}>
                                    {testing
                                        ? <><span className="ocr-spinner" /> Reading…</>
                                        : <>📷 Choose test image</>}
                                    <input type="file" accept="image/*" hidden
                                        disabled={testing} onChange={handleTestImage} />
                                </label>
                                {testResult && (
                                    <div className="ocr-test-result">
                                        <div className="ocr-test-text">
                                            <small>Model read ({testResult.num_found} characters):</small>
                                            <span>{testResult.text}</span>
                                        </div>
                                        {testResult.preview && (
                                            <img className="ocr-test-preview" src={testResult.preview}
                                                alt="detected characters" />
                                        )}
                                        <div className="ocr-perclass">
                                            {testResult.characters.map((c, i) => (
                                                <span key={i}
                                                    className={`ocr-chip ${c.confidence >= 0.8 ? 'ok' : c.confidence >= 0.5 ? 'warn' : 'bad'}`}>
                                                    {c.char} {(c.confidence * 100).toFixed(0)}%
                                                </span>
                                            ))}
                                        </div>
                                        <p className="ocr-hint">
                                            Green = confident, yellow = unsure, red = probably wrong.
                                            Wrong characters? Label more examples of them and retrain/fine-tune.
                                        </p>
                                    </div>
                                )}
                            </div>

                            <p className="ocr-hint">
                                Bundle <b>ocr_model.tflite</b> + <b>labels.txt</b> into the Digi OCR app
                                (tflite_flutter). Input: one grayscale character crop, resized to
                                {` ${finalMeta?.img_size || imgSize}×${finalMeta?.img_size || imgSize}`}, values scaled to 0–1.
                            </p>
                        </div>
                    )}

                    {/* ── Tesseract results / trained model ─────── */}
                    {engine === 'tesseract' && (result?.engine === 'tesseract' || modelInfo?.tesseract?.has_model) && (
                        <div className="ocr-section">
                            <h3>
                                {result?.engine === 'tesseract'
                                    ? <><Check size={15} className="ocr-ok" /> Fine-tuning complete</>
                                    : 'Fine-tuned Tesseract model'}
                            </h3>
                            {(() => {
                                const tm = result?.engine === 'tesseract' ? result : modelInfo?.tesseract?.meta;
                                if (!tm) return null;
                                return (
                                    <div className="ocr-results">
                                        <div className="ocr-result-big">
                                            <span>{pct(tm.val_accuracy)}</span>
                                            <small>character accuracy on held-out lines
                                                ({tm.val_lines} lines; trained on {tm.train_lines})</small>
                                        </div>
                                        {tm.top_confusions?.length > 0 && (
                                            <p className="ocr-hint">
                                                Most confused: {tm.top_confusions.slice(0, 5)
                                                    .map(t => `${t.pair.replace('->', ' → ')} (${t.count})`).join(', ')}.
                                                Label more examples of these characters and retrain.
                                            </p>
                                        )}
                                        {(tm.eval_lines || []).slice(0, 8).map((l, i) => (
                                            <p key={i} className="ocr-hint" style={{ margin: '2px 0' }}>
                                                <b>{l.truth}</b> → read as <b>{l.predicted || '(nothing)'}</b>
                                                {l.truth === l.predicted ? ' ✓' : ''}
                                            </p>
                                        ))}
                                    </div>
                                );
                            })()}
                            <div className="ocr-downloads">
                                <button onClick={() => download('traineddata', 'engraved.traineddata')}>
                                    <Download size={14} /> engraved.traineddata
                                </button>
                                <button onClick={() => download('tess_meta', 'tess_meta.json')}>
                                    <Download size={14} /> tess_meta.json
                                </button>
                            </div>
                            {/* ── Test window ─────────────────── */}
                            <div className="ocr-test">
                                <h3>Test the model</h3>
                                <p className="ocr-hint">
                                    Upload a plate photo — text lines are found and read with the
                                    fine-tuned Tesseract model. Use a photo it was NOT trained on.
                                </p>
                                <label className={`ocr-btn-test ${testing ? 'busy' : ''}`}>
                                    {testing
                                        ? <><span className="ocr-spinner" /> Reading…</>
                                        : <>📷 Choose test image</>}
                                    <input type="file" accept="image/*" hidden
                                        disabled={testing} onChange={handleTestImage} />
                                </label>
                                {testResult && (
                                    <div className="ocr-test-result">
                                        <div className="ocr-test-text">
                                            <small>Model read:</small>
                                            <span>{testResult.text || '(nothing)'}</span>
                                        </div>
                                        {testResult.preview && (
                                            <img className="ocr-test-preview" src={testResult.preview}
                                                alt="detected text lines" />
                                        )}
                                    </div>
                                )}
                            </div>
                            <p className="ocr-hint">
                                Bundle <b>engraved.traineddata</b> into the Digi OCR app with the
                                flutter_tesseract_ocr plugin (place it in the app's tessdata folder
                                and use language code <b>engraved</b>).
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default OcrTrainingPanel;
