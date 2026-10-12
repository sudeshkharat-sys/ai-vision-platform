import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { Sparkles, X, Upload, Play, Square, Send } from 'lucide-react';
import './VideoPanel.css';
import { API_URL } from '../config';

const POLL_MS = 2000;

// Label this project's videos with a detector .pt on this PC, review the result
// in Review Annotation, then copy the reviewed frames to a training project.
export default function AutoLabelPanel({ project, onClose, onImagesChanged }) {
    const [info, setInfo] = useState({ running: false, model: null, status: null });
    const [fps, setFps] = useState(3);
    const [conf, setConf] = useState(0.25);
    const [window_, setWindow] = useState(true);
    const [dedupe, setDedupe] = useState(true);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const [projects, setProjects] = useState([]);
    const [target, setTarget] = useState('');
    const [newName, setNewName] = useState('');
    const [sent, setSent] = useState(null);
    const fileRef = useRef(null);

    const refresh = useCallback(async () => {
        try {
            const res = await axios.get(`${API_URL}/autolabel/${project.id}/status`);
            setInfo(res.data);
        } catch { /* ignore */ }
    }, [project.id]);

    useEffect(() => {
        refresh();
        axios.get(`${API_URL}/projects`).then(r => setProjects(r.data.filter(p => p.id !== project.id))).catch(() => {});
    }, [project.id, refresh]);

    useEffect(() => {
        if (!info.running) return undefined;
        const t = setInterval(refresh, POLL_MS);
        return () => clearInterval(t);
    }, [info.running, refresh]);

    // when a run ends, let the workspace reload its image list
    const wasRunning = useRef(false);
    useEffect(() => {
        if (wasRunning.current && !info.running) onImagesChanged?.();
        wasRunning.current = info.running;
    }, [info.running, onImagesChanged]);

    const fail = (e, fallback) => setError(e?.response?.data?.detail || fallback);

    const upload = async (file) => {
        if (!file) return;
        setBusy(true); setError('');
        try {
            const fd = new FormData();
            fd.append('file', file);
            await axios.post(`${API_URL}/autolabel/${project.id}/model`, fd);
            await refresh();
        } catch (e) { fail(e, 'Upload failed'); }
        setBusy(false);
    };

    const start = async () => {
        setBusy(true); setError(''); setSent(null);
        try {
            await axios.post(`${API_URL}/autolabel/${project.id}/run`, { fps: Number(fps), conf: Number(conf), window: window_, dedupe });
            setInfo(i => ({ ...i, running: true }));
            setTimeout(refresh, 800);
        } catch (e) { fail(e, 'Could not start'); }
        setBusy(false);
    };

    const stop = async () => {
        try { await axios.post(`${API_URL}/autolabel/${project.id}/stop`); } catch { /* ignore */ }
    };

    const send = async () => {
        setBusy(true); setError(''); setSent(null);
        try {
            const res = await axios.post(`${API_URL}/autolabel/${project.id}/send`,
                target ? { target_project_id: target } : { new_name: newName });
            setSent(res.data);
        } catch (e) { fail(e, 'Could not send'); }
        setBusy(false);
    };

    const st = info.status;
    return (
        <div className="vp-overlay" onClick={onClose}>
            <div className="vp-modal" onClick={e => e.stopPropagation()}>
                <div className="vp-header">
                    <div className="vp-header-left">
                        <span className="vp-header-icon"><Sparkles size={20} /></span>
                        <div>
                            <h2 className="vp-title">Auto-label videos</h2>
                            <p className="vp-subtitle">Detector .pt on this PC → boxes on video frames → Review → training project</p>
                        </div>
                    </div>
                    <button className="vp-close" onClick={onClose}><X size={18} /></button>
                </div>

                <div style={{ padding: 18, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
                    <section>
                        <b>1. Detector model (.pt)</b>
                        <p style={{ margin: '4px 0 8px', color: '#666', fontSize: 13 }}>
                            Upload your videos with <i>Import Video</i> first (no need to extract frames).
                        </p>
                        <input ref={fileRef} type="file" accept=".pt" style={{ display: 'none' }}
                               onChange={e => { upload(e.target.files[0]); e.target.value = ''; }} />
                        <button className="btn-action" disabled={busy || info.running} onClick={() => fileRef.current.click()}>
                            <Upload size={14} /> {info.model ? 'Replace .pt' : 'Upload .pt'}
                        </button>
                        {info.model && (
                            <p style={{ fontSize: 13, marginTop: 8 }}>
                                <b>{info.model.filename}</b> — classes: {info.model.classes.join(', ')}
                            </p>
                        )}
                    </section>

                    <section>
                        <b>2. Run</b>
                        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', margin: '8px 0' }}>
                            <label>Frames/sec <input type="number" min="0.5" step="0.5" value={fps} style={{ width: 60 }}
                                                    onChange={e => setFps(e.target.value)} /></label>
                            <label>Min confidence <input type="number" min="0.05" max="0.95" step="0.05" value={conf}
                                                         style={{ width: 60 }} onChange={e => setConf(e.target.value)} /></label>
                            <label><input type="checkbox" checked={window_} onChange={e => setWindow(e.target.checked)} /> Show preview window</label>
                            <label><input type="checkbox" checked={dedupe} onChange={e => setDedupe(e.target.checked)} /> Skip near-identical frames</label>
                        </div>
                        {!info.running ? (
                            <button className="btn-action" disabled={busy || !info.model} onClick={start}>
                                <Play size={14} /> Start auto-label
                            </button>
                        ) : (
                            <button className="btn-action" onClick={stop}><Square size={14} /> Stop</button>
                        )}
                        <p style={{ fontSize: 12, color: '#777', marginTop: 6 }}>
                            Preview window: q = stop, space = pause. Only frames where something is detected are saved.
                        </p>
                        {st && (
                            <div style={{ fontSize: 13, marginTop: 8 }}>
                                <div><b>{st.state}</b>{st.video ? ` — ${st.video}` : ''} ({st.videos_done}/{st.videos_total} videos)</div>
                                <div>Frames looked at: {st.frames_seen} · saved: {st.saved} · boxes: {st.boxes}</div>
                                {st.message && <div style={{ color: '#b45309' }}>{st.message}</div>}
                                {Object.entries(st.per_video || {}).map(([v, c]) => (
                                    <div key={v} style={{ color: '#555' }}>{v}: {Object.entries(c).map(([k, n]) => `${k} ${n}`).join(', ') || 'nothing found'}</div>
                                ))}
                            </div>
                        )}
                    </section>

                    <section>
                        <b>3. Review</b>
                        <p style={{ margin: '4px 0', fontSize: 13, color: '#666' }}>
                            Close this window → <i>Review Annotation</i> → tick <i>Auto-annotated only</i>. Accept or fix the boxes
                            (a frame counts as reviewed once none of its boxes are still machine boxes). Locked / unlocked comes from
                            <i> Data Segments</i> (rules on which classes a frame has), not from the video name.
                        </p>
                    </section>

                    <section>
                        <b>4. Send reviewed frames to a training project</b>
                        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', margin: '8px 0' }}>
                            <select value={target} onChange={e => setTarget(e.target.value)}>
                                <option value="">— new project —</option>
                                {projects.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                            </select>
                            {!target && <input placeholder="New project name" value={newName} onChange={e => setNewName(e.target.value)} />}
                            <button className="btn-action" disabled={busy || info.running || (!target && !newName.trim())} onClick={send}>
                                <Send size={14} /> Send reviewed frames
                            </button>
                        </div>
                        {sent && (
                            <p style={{ fontSize: 13 }}>
                                Copied {sent.copied} frames to “{sent.target_name}”. Left behind (still have machine boxes): {sent.skipped_unreviewed}
                                {sent.missing_files ? ` · missing files: ${sent.missing_files}` : ''}.
                            </p>
                        )}
                    </section>

                    {error && <div style={{ color: '#b91c1c', fontSize: 13 }}>{error}</div>}
                </div>
            </div>
        </div>
    );
}
