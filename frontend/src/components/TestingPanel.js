import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { API_URL, BASE_URL } from '../config';
import './TestingPanel.css';

const SOURCE_LABELS = {
    detector: 'Detector (project Main / Seed)',
    segmentation: 'Segmentation model',
    classifier: 'Classifier (crop + classify)',
    uploaded: 'Uploaded .pt',
};
const pct = (v) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`);
const errMsg = (e) => e?.response?.data?.detail || e?.message || 'Request failed';

/** Auth-protected files can't be a plain <img src>; fetch as blob. */
function BlobImg({ url, alt }) {
    const [src, setSrc] = useState(null);
    useEffect(() => {
        let dead = false, obj = null;
        axios.get(url, { responseType: 'blob' }).then(r => {
            if (dead) return;
            obj = URL.createObjectURL(r.data);
            setSrc(obj);
        }).catch(() => { });
        return () => { dead = true; if (obj) URL.revokeObjectURL(obj); };
    }, [url]);
    return src ? <img src={src} alt={alt} loading="lazy" /> : <div style={{ aspectRatio: '4/3', background: '#eee' }} />;
}

function useSources(projectId) {
    const [sources, setSources] = useState(null);
    const reload = useCallback(() => {
        if (!projectId) return;
        axios.get(`${API_URL}/testing/${projectId}/sources`).then(r => setSources(r.data)).catch(() => setSources(null));
    }, [projectId]);
    useEffect(() => { setSources(null); reload(); }, [reload]);
    return [sources, reload];
}

function usePolling(url, enabled, onData, ms = 1000) {
    const cb = useRef(onData);
    cb.current = onData;
    useEffect(() => {
        if (!enabled || !url) return undefined;
        let dead = false;
        const tick = () => axios.get(url).then(r => { if (!dead) cb.current(r.data); }).catch(() => { });
        tick();
        const t = setInterval(tick, ms);
        return () => { dead = true; clearInterval(t); };
    }, [url, enabled, ms]);
}

/* ── Source picker shared by all three modes ─────────────────── */
function SourceBar({ projectId, sources, reload, value, onChange }) {
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState(null);
    const upload = async (file) => {
        if (!file) return;
        setBusy(true); setErr(null);
        try {
            const fd = new FormData();
            fd.append('file', file);
            await axios.post(`${API_URL}/testing/${projectId}/model`, fd);
            await reload();
            onChange({ ...value, source: 'uploaded' });
        } catch (e) { setErr(errMsg(e)); } finally { setBusy(false); }
    };
    const avail = (k) => sources?.[k]?.available;
    return (
        <div className="tp-card">
            <h3>Model to test</h3>
            <p className="tp-sub">Pick a model trained in this project, or upload any YOLO detection / segmentation .pt.</p>
            <div className="tp-row">
                <label className="tp-field">Model
                    <select value={value.source} onChange={e => onChange({ ...value, source: e.target.value })}>
                        {Object.keys(SOURCE_LABELS).map(k => (
                            <option key={k} value={k} disabled={!avail(k)}>
                                {SOURCE_LABELS[k]}{avail(k) ? '' : ' — not available'}
                            </option>
                        ))}
                    </select>
                </label>
                <label className="tp-field">{value.source === 'classifier' ? 'Region confidence' : 'Confidence'}
                    <input type="number" min="0.01" max="0.99" step="0.05" value={value.conf}
                        onChange={e => onChange({ ...value, conf: Number(e.target.value) })} />
                </label>
                {value.source === 'classifier' && (
                    <label className="tp-field">Classifier threshold
                        <input type="number" min="0.05" max="0.99" step="0.05" value={value.cls_threshold}
                            onChange={e => onChange({ ...value, cls_threshold: Number(e.target.value) })} />
                    </label>
                )}
                {value.source === 'uploaded' && (
                    <label className="tp-field" title="Tick if the model was trained on CLAHE-preprocessed images">Preprocess
                        <select value={value.preprocess ? '1' : '0'} onChange={e => onChange({ ...value, preprocess: e.target.value === '1' })}>
                            <option value="0">Off (default)</option><option value="1">On (CLAHE)</option>
                        </select>
                    </label>
                )}
                <label className="tp-field">Upload .pt
                    <input type="file" accept=".pt" disabled={busy} onChange={e => upload(e.target.files[0])} />
                </label>
            </div>
            {sources?.uploaded?.meta && (
                <p className="tp-note">Uploaded: {sources.uploaded.meta.filename} ({sources.uploaded.meta.task}, {sources.uploaded.meta.classes.length} classes)</p>
            )}
            {err && <div className="tp-err">{err}</div>}
        </div>
    );
}

const qs = (o) => `source=${o.source}&conf=${o.conf}&preprocess=${o.preprocess}&cls_threshold=${o.cls_threshold}`;

/* ── Mode 1: upload images ───────────────────────────────────── */
function ImageTest({ projectId, opts }) {
    const [results, setResults] = useState([]);
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState(null);
    const run = async (files) => {
        setBusy(true); setErr(null); setResults([]);
        for (const f of Array.from(files)) {
            try {
                const fd = new FormData();
                fd.append('file', f);
                const r = await axios.post(`${API_URL}/testing/${projectId}/image?${qs(opts)}`, fd);
                setResults(prev => [...prev, { name: f.name, ...r.data }]);
            } catch (e) { setErr(`${f.name}: ${errMsg(e)}`); break; }
        }
        setBusy(false);
    };
    return (
        <div className="tp-card">
            <h3>Test on images</h3>
            <p className="tp-sub">Upload one or many images — you get the picture back with the model's boxes drawn.</p>
            <input type="file" accept="image/*" multiple disabled={busy} onChange={e => e.target.files.length && run(e.target.files)} />
            {busy && <p className="tp-note">Running…</p>}
            {err && <div className="tp-err">{err}</div>}
            <div className="tp-results">
                {results.map((r, i) => (
                    <div key={i} className="tp-result">
                        <b>{r.name}</b>{' '}
                        {r.kind === 'classifier'
                            ? <span className="tp-note">{r.status}{r.state ? ` → ${r.state} (${pct(r.confidence)})` : ''}</span>
                            : <span className="tp-note">{r.detections.length} detection(s): {Object.entries(r.detections.reduce((a, d) => ({ ...a, [d.class_name]: (a[d.class_name] || 0) + 1 }), {})).map(([k, v]) => `${k}×${v}`).join(', ') || 'none'}</span>}
                        <div><img src={r.preview} alt={r.name} /></div>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ── Mode 2: upload a video ──────────────────────────────────── */
function VideoTest({ projectId, opts }) {
    const [fps, setFps] = useState(5);
    const [flagBelow, setFlagBelow] = useState(0.4);
    const [st, setSt] = useState(null);
    const [err, setErr] = useState(null);
    const running = st?.state === 'running';
    usePolling(`${API_URL}/testing/${projectId}/video/status`, true, setSt, 1000);

    const start = async (file) => {
        if (!file) return;
        setErr(null);
        try {
            const fd = new FormData();
            fd.append('file', file);
            await axios.post(`${API_URL}/testing/${projectId}/video?${qs(opts)}&fps=${fps}&flag_below=${flagBelow}`, fd);
            setSt({ state: 'running', frames: 0, total: 0, flagged: [] });
        } catch (e) { setErr(errMsg(e)); }
    };
    const download = async () => {
        const r = await axios.get(`${API_URL}/testing/${projectId}/video/output`, { responseType: 'blob' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(r.data);
        a.download = 'tested_video.mp4';
        a.click();
    };
    const pctDone = st?.total ? Math.min(100, (st.frames / st.total) * 100) : 0;
    return (
        <div className="tp-card">
            <h3>Test on a video</h3>
            <p className="tp-sub">The video is analysed at the chosen frames per second. You get an annotated video, and the frames where nothing was found / confidence was low are collected below.</p>
            <div className="tp-row">
                <label className="tp-field">Frames / second
                    <input type="number" min="0.5" max="30" step="0.5" value={fps} onChange={e => setFps(Number(e.target.value))} />
                </label>
                <label className="tp-field" title="Flag a frame when the best box is below this confidence">Flag below conf
                    <input type="number" min="0.05" max="0.99" step="0.05" value={flagBelow} onChange={e => setFlagBelow(Number(e.target.value))} />
                </label>
                <label className="tp-field">Video file
                    <input type="file" accept="video/*" disabled={running} onChange={e => start(e.target.files[0])} />
                </label>
                {running && <button className="tp-btn tp-btn--ghost" onClick={() => axios.post(`${API_URL}/testing/${projectId}/video/stop`)}>Stop</button>}
            </div>
            {err && <div className="tp-err">{err}</div>}
            {st && st.state !== 'idle' && (
                <>
                    <div className="tp-progress"><div style={{ width: `${st.state === 'done' ? 100 : pctDone}%` }} /></div>
                    <p className="tp-note">
                        {st.state}{st.message ? `: ${st.message}` : ''} · {st.frames} frames analysed · {st.flagged?.length || 0} flagged
                        {st.no_detection ? ` (${st.no_detection} with no detection)` : ''}
                    </p>
                    {st.has_output && !running && <button className="tp-btn" onClick={download}>Download annotated video</button>}
                    {st.flagged?.length > 0 && (
                        <>
                            <h3 style={{ marginTop: 14 }}>Flagged frames</h3>
                            <div className="tp-grid">
                                {st.flagged.slice(0, 60).map(f => (
                                    <div key={f.name} className="tp-thumb tp-thumb--bad">
                                        <BlobImg url={`${API_URL}/testing/${projectId}/video/flagged/${f.name}`} alt={f.name} />
                                        <div><span className="tp-tag">{f.reason}</span>{f.time_s}s</div>
                                    </div>
                                ))}
                            </div>
                            {st.flagged.length > 60 && <p className="tp-note">Showing the first 60 of {st.flagged.length}.</p>}
                        </>
                    )}
                </>
            )}
        </div>
    );
}

/* ── Failed-image viewer with GT vs prediction ───────────────── */
function Viewer({ row, source, onClose }) {
    const boxes = (list, color, dash) => list.map((b, i) => {
        const [xc, yc, w, h] = b.bbox;
        return (
            <g key={color + i}>
                <rect x={xc - w / 2} y={yc - h / 2} width={w} height={h} fill="none" stroke={color} strokeWidth="2"
                    strokeDasharray={dash} vectorEffect="non-scaling-stroke" />
            </g>
        );
    });
    const label = (list, color, dy, pre) => list.map((b, i) => (
        <text key={color + 't' + i} x={b.bbox[0] - b.bbox[2] / 2} y={b.bbox[1] - b.bbox[3] / 2 - 0.004 + dy}
            fontSize="0.022" fill={color}>{pre}{b.class_name}{b.conf != null ? ` ${b.conf.toFixed(2)}` : ''}</text>
    ));
    return (
        <div className="tp-modal" onClick={onClose}>
            <div className="tp-modal-box" onClick={e => e.stopPropagation()}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <b>{row.filename}</b><button className="tp-btn tp-btn--ghost" onClick={onClose}>Close</button>
                </div>
                {source === 'classifier'
                    ? <p className="tp-legend">Expected <b>{row.expected}</b> · predicted <b>{row.predicted || row.status}</b> ({pct(row.confidence)}) · {row.reason}</p>
                    : <p className="tp-legend">
                        <span style={{ color: '#1a9d55' }}>■ your label (dashed)</span>{'  '}
                        <span style={{ color: '#dc143c' }}>■ model prediction</span>{'  '}
                        — missed {row.missed}, wrong class {row.wrong_class}, extra {row.extra}
                    </p>}
                <div className="tp-overlay">
                    <img src={`${BASE_URL}${row.filepath}`} alt={row.filename} />
                    {source !== 'classifier' && (
                        <svg viewBox="0 0 1 1" preserveAspectRatio="none">
                            {boxes(row.gts, '#1a9d55', '0.02 0.01')}
                            {boxes(row.preds, '#dc143c')}
                            {label(row.gts, '#1a9d55', 0, '')}
                            {label(row.preds, '#dc143c', 0.026, '')}
                        </svg>
                    )}
                </div>
            </div>
        </div>
    );
}

/* ── Mode 3: test on the project's own training data ─────────── */
function DataTest({ projectId, projectName, opts, sourceReady }) {
    const [iou, setIou] = useState(0.5);
    const [allowExtra, setAllowExtra] = useState(false);
    const [limit, setLimit] = useState('');
    const [st, setSt] = useState(null);
    const [data, setData] = useState(null);
    const [view, setView] = useState('failed');
    const [open, setOpen] = useState(null);
    const [err, setErr] = useState(null);
    const [newName, setNewName] = useState('');
    const [sent, setSent] = useState(null);
    const [sending, setSending] = useState(false);
    const running = st?.state === 'running';

    const loadResults = useCallback(() => {
        axios.get(`${API_URL}/testing/${projectId}/training-data/results`).then(r => setData(r.data)).catch(() => setData(null));
    }, [projectId]);
    useEffect(() => { setData(null); setSent(null); setSt(null); loadResults(); }, [projectId, loadResults]);
    useEffect(() => { setNewName(`${projectName} - failed`); }, [projectName]);

    usePolling(`${API_URL}/testing/${projectId}/training-data/status`, true, (d) => {
        setSt(prev => { if (prev?.state === 'running' && d.state !== 'running') loadResults(); return d; });
    }, 1000);

    const start = async () => {
        setErr(null); setSent(null);
        try {
            await axios.post(`${API_URL}/testing/${projectId}/training-data/run`, {
                source: opts.source, conf: opts.conf, iou, allow_extra: allowExtra, preprocess: opts.preprocess,
                cls_threshold: opts.cls_threshold, limit: limit === '' ? null : Number(limit),
            });
            setData(null);
            setSt({ state: 'running', done: 0, total: 1, failed: 0 });
        } catch (e) { setErr(errMsg(e)); }
    };
    const send = async () => {
        setSending(true); setErr(null);
        try {
            const r = await axios.post(`${API_URL}/testing/${projectId}/training-data/send-failed`, { new_name: newName });
            setSent(r.data);
        } catch (e) { setErr(errMsg(e)); } finally { setSending(false); }
    };

    const s = data?.summary;
    const rows = (data?.rows || []).filter(r => view === 'all' || !r.passed);
    const isCls = data?.source === 'classifier';
    return (
        <div className="tp-card">
            <h3>Test on training data</h3>
            <p className="tp-sub">Runs the model on this project's labelled images and compares with your labels. An image <b>fails</b> if a labelled object is missed or gets the wrong class{allowExtra ? '' : ', or the model draws an extra box'}. Failed images can then be sent to a new project for another training round.</p>
            <div className="tp-row">
                {opts.source !== 'classifier' && (
                    <label className="tp-field" title="A box counts as found when it overlaps your label at least this much">Match IoU
                        <input type="number" min="0.1" max="0.95" step="0.05" value={iou} onChange={e => setIou(Number(e.target.value))} />
                    </label>
                )}
                {opts.source !== 'classifier' && (
                    <label className="tp-field">Extra boxes
                        <select value={allowExtra ? '1' : '0'} onChange={e => setAllowExtra(e.target.value === '1')}>
                            <option value="0">Count as failure</option><option value="1">Ignore</option>
                        </select>
                    </label>
                )}
                <label className="tp-field" title="Quick check on the first N images">Only first N
                    <input type="number" min="1" placeholder="all" value={limit} onChange={e => setLimit(e.target.value)} />
                </label>
                <button className="tp-btn" disabled={running || !sourceReady} onClick={start}>Run test</button>
                {running && <button className="tp-btn tp-btn--ghost" onClick={() => axios.post(`${API_URL}/testing/${projectId}/training-data/stop`)}>Stop</button>}
            </div>
            {opts.source === 'classifier' && <p className="tp-note">The classifier is tested against your Data Segments labels (Training Hub → Data Segments).</p>}
            {err && <div className="tp-err" style={{ marginTop: 8 }}>{err}</div>}
            {running && (
                <>
                    <div className="tp-progress"><div style={{ width: `${(st.done / Math.max(1, st.total)) * 100}%` }} /></div>
                    <p className="tp-note">{st.done} / {st.total} images · {st.failed} failed so far</p>
                </>
            )}
            {st?.state === 'failed' && <div className="tp-err">{st.message}</div>}

            {s && (
                <>
                    <div className="tp-tiles">
                        <div className="tp-tile"><b>{s.images}</b><span>images</span></div>
                        <div className="tp-tile tp-tile--good"><b>{s.passed}</b><span>passed</span></div>
                        <div className="tp-tile tp-tile--bad"><b>{s.failed}</b><span>failed</span></div>
                        <div className="tp-tile"><b>{pct(s.pass_rate)}</b><span>pass rate</span></div>
                        {!isCls && <>
                            <div className="tp-tile"><b>{pct(s.recall)}</b><span>recall</span></div>
                            <div className="tp-tile"><b>{pct(s.precision)}</b><span>precision</span></div>
                            <div className="tp-tile"><b>{s.missed}</b><span>missed</span></div>
                            <div className="tp-tile"><b>{s.wrong_class}</b><span>wrong class</span></div>
                            <div className="tp-tile"><b>{s.extra}</b><span>extra</span></div>
                        </>}
                    </div>
                    {data.stopped && <p className="tp-note">Stopped early — only the images tested so far are counted.</p>}

                    {s.failed > 0 && (
                        <div className="tp-row" style={{ background: '#fafafa', padding: 12, borderRadius: 10 }}>
                            <label className="tp-field">Send the {s.failed} failed images (with labels) to a new project
                                <input style={{ width: 280 }} value={newName} onChange={e => setNewName(e.target.value)} />
                            </label>
                            <button className="tp-btn" disabled={sending || !newName.trim()} onClick={send}>Create project &amp; send</button>
                        </div>
                    )}
                    {sent && (
                        <p className="tp-note" style={{ color: '#1a9d55' }}>
                            Created “{sent.target_name}” with {sent.copied} images{sent.missing_files ? ` (${sent.missing_files} files missing)` : ''}.
                            {sent.weights ? ` The tested model is saved there as custom weights “${sent.weights}” — pick it when training to continue from it.` : ''}
                            {' '}Open it from the Training panel tab.
                        </p>
                    )}

                    <div className="tp-tabs" style={{ marginTop: 12 }}>
                        <button className={`tp-tab ${view === 'failed' ? 'tp-tab--on' : ''}`} onClick={() => setView('failed')}>Failed ({s.failed})</button>
                        <button className={`tp-tab ${view === 'all' ? 'tp-tab--on' : ''}`} onClick={() => setView('all')}>All ({s.images})</button>
                    </div>
                    <div className="tp-grid">
                        {rows.slice(0, 200).map(r => (
                            <div key={r.id} className={`tp-thumb ${r.passed ? '' : 'tp-thumb--bad'}`} onClick={() => setOpen(r)}>
                                <img src={`${BASE_URL}${r.filepath}`} alt={r.filename} loading="lazy" />
                                <div>
                                    <div style={{ padding: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.filename}</div>
                                    {isCls
                                        ? (r.passed ? 'ok' : <span className="tp-tag">{r.reason}</span>)
                                        : r.passed ? 'ok' : <>
                                            {r.missed > 0 && <span className="tp-tag">missed {r.missed}</span>}
                                            {r.wrong_class > 0 && <span className="tp-tag">wrong {r.wrong_class}</span>}
                                            {r.extra > 0 && <span className="tp-tag">extra {r.extra}</span>}
                                            {r.error && <span className="tp-tag">{r.error}</span>}
                                        </>}
                                </div>
                            </div>
                        ))}
                    </div>
                    {rows.length > 200 && <p className="tp-note">Showing the first 200 of {rows.length}.</p>}
                    {rows.length === 0 && <p className="tp-note">{view === 'failed' ? 'No failed images 🎉' : 'Nothing to show'}</p>}
                </>
            )}
            {open && <Viewer row={open} source={data?.source} onClose={() => setOpen(null)} />}
        </div>
    );
}

/* ── Panel ───────────────────────────────────────────────────── */
export default function TestingPanel({ projects }) {
    const [projectId, setProjectId] = useState(() => {
        try { return sessionStorage.getItem('testing_project') || ''; } catch { return ''; }
    });
    const [mode, setMode] = useState('data');
    const [opts, setOpts] = useState({ source: 'detector', conf: 0.25, preprocess: false, cls_threshold: 0.6 });
    const [sources, reload] = useSources(projectId);

    useEffect(() => {
        if (!projectId && projects.length) setProjectId(projects[0].id);
    }, [projects, projectId]);
    useEffect(() => {
        try { if (projectId) sessionStorage.setItem('testing_project', projectId); } catch { /* ignore */ }
    }, [projectId]);
    // fall back to a source that exists for the chosen project
    useEffect(() => {
        if (!sources || sources[opts.source]?.available) return;
        const first = ['detector', 'segmentation', 'classifier', 'uploaded'].find(k => sources[k]?.available);
        if (first) setOpts(o => ({ ...o, source: first }));
    }, [sources, opts.source]);

    const project = projects.find(p => p.id === projectId);
    const ready = !!sources?.[opts.source]?.available;
    return (
        <div className="tp-root">
            <div className="tp-card">
                <h3>Project</h3>
                <p className="tp-sub">Choose the project whose model you want to test.</p>
                <label className="tp-field">Project
                    <select value={projectId} onChange={e => setProjectId(e.target.value)}>
                        {projects.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                    </select>
                </label>
            </div>
            {projectId && (
                <>
                    <SourceBar projectId={projectId} sources={sources} reload={reload} value={opts} onChange={setOpts} />
                    {!ready && sources && <div className="tp-err">No usable model for “{SOURCE_LABELS[opts.source]}”. Train one in the Training panel or upload a .pt.</div>}
                    <div className="tp-tabs">
                        {[['data', 'Training data'], ['images', 'Upload images'], ['video', 'Upload video']].map(([k, l]) => (
                            <button key={k} className={`tp-tab ${mode === k ? 'tp-tab--on' : ''}`} onClick={() => setMode(k)}>{l}</button>
                        ))}
                    </div>
                    {mode === 'data' && <DataTest projectId={projectId} projectName={project?.name || ''} opts={opts} sourceReady={ready} />}
                    {mode === 'images' && <ImageTest projectId={projectId} opts={opts} />}
                    {mode === 'video' && <VideoTest projectId={projectId} opts={opts} />}
                </>
            )}
        </div>
    );
}
