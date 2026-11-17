import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import './TestingPanel.css';

const errMsg = (e) => e?.response?.data?.detail || e?.message || 'Request failed';
const PAGE = 30;

/** Crops are auth-protected, so fetch as blob. */
function Crop({ url }) {
    const [src, setSrc] = useState(null);
    useEffect(() => {
        let dead = false, obj = null;
        setSrc(null);
        axios.get(url, { responseType: 'blob' }).then(r => {
            if (dead) return;
            obj = URL.createObjectURL(r.data);
            setSrc(obj);
        }).catch(() => { });
        return () => { dead = true; if (obj) URL.revokeObjectURL(obj); };
    }, [url]);
    return src
        ? <img src={src} alt="" style={{ width: '100%', display: 'block', aspectRatio: '1/1', objectFit: 'contain', background: '#111' }} />
        : <div style={{ aspectRatio: '1/1', background: '#222' }} />;
}

/**
 * Rectify: re-check auto-labelled boxes that are unusually dark (torch light
 * missed the lock). Self-contained; talks only to /rectify/*.
 */
export default function RectifyTab({ projectId }) {
    const base = `${API_URL}/rectify/${projectId}`;
    const [states, setStates] = useState([]);
    const [state, setState] = useState('');
    const [onlyAuto, setOnlyAuto] = useState(true);
    const [st, setSt] = useState({ state: 'idle' });
    const [data, setData] = useState({ summary: {}, items: [] });
    const [outliersOnly, setOutliersOnly] = useState(true);
    const [enhance, setEnhance] = useState(true);
    const [page, setPage] = useState(0);
    const [done, setDone] = useState({});     // ann_id -> 'kept' | new state
    const [err, setErr] = useState(null);
    const timer = useRef(null);

    const loadStates = useCallback(() => {
        axios.get(`${base}/states`).then(r => {
            setStates(r.data);
            // Saved boxes become 'manual', so a project can have no 'auto' boxes left.
            if (!r.data.some(x => x.source === 'auto')) setOnlyAuto(false);
        }).catch(e => setErr(errMsg(e)));
    }, [base]);

    const loadResults = useCallback(() => {
        axios.get(`${base}/results`).then(r => { setData(r.data); setPage(0); setDone({}); }).catch(() => { });
    }, [base]);

    useEffect(() => { setData({ summary: {}, items: [] }); setSt({ state: 'idle' }); setState(''); loadStates();
        axios.get(`${base}/status`).then(r => { setSt(r.data); if (r.data.state === 'done') loadResults(); }).catch(() => { });
    }, [projectId]); // eslint-disable-line

    useEffect(() => {
        if (st.state !== 'running') return undefined;
        timer.current = setInterval(() => {
            axios.get(`${base}/status`).then(r => {
                setSt(r.data);
                if (r.data.state === 'done') loadResults();
            }).catch(() => { });
        }, 1000);
        return () => clearInterval(timer.current);
    }, [st.state, base, loadResults]);

    const start = async () => {
        setErr(null);
        try {
            await axios.post(`${base}/scan`, { state: state || null, source: onlyAuto ? 'auto' : 'all' });
            setSt({ state: 'running', done: 0, total: 1 });
        } catch (e) { setErr(errMsg(e)); }
    };

    const act = async (path, body, label, annId) => {
        try {
            await axios.post(`${base}/${path}`, body);
            setDone(d => ({ ...d, [annId]: label }));
        } catch (e) { setErr(errMsg(e)); }
    };
    const undo = async (annId) => {
        try {
            await axios.post(`${base}/undo`, { ann_id: annId });
            setDone(d => { const n = { ...d }; delete n[annId]; return n; });
        } catch (e) { setErr(errMsg(e)); }
    };

    const stateNames = [...new Set(states.map(s => s.state))];
    const items = outliersOnly ? data.items.filter(i => i.outlier) : data.items;
    const pageItems = items.slice(page * PAGE, page * PAGE + PAGE);
    const pages = Math.max(1, Math.ceil(items.length / PAGE));
    const sm = data.summary || {};
    const running = st.state === 'running';
    const reviewed = Object.keys(done).length;

    return (
        <div className="tp-card">
            <h3>Rectify labels</h3>
            <p className="tp-sub">Finds auto-labelled boxes that are much darker than normal (e.g. a lock "closed" in an area the torch did not light) so you can re-check them first. Nothing changes until you click a button; every change can be undone.</p>
            {err && <div className="tp-err" onClick={() => setErr(null)}>{err}</div>}

            <div className="tp-row">
                <label className="tp-field">Label to check
                    <select value={state} onChange={e => setState(e.target.value)}>
                        <option value="">All labels</option>
                        {stateNames.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </label>
                <label className="tp-field" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                    <input type="checkbox" checked={onlyAuto} onChange={e => setOnlyAuto(e.target.checked)} style={{ minWidth: 0 }} />
                    Only auto-labelled
                </label>
                <button className="tp-btn" disabled={running} onClick={start}>{running ? 'Scanning…' : 'Scan darkness'}</button>
            </div>
            {states.length > 0 && !states.some(x => x.source === 'auto') && <p className="tp-note" style={{ marginTop: 8, color: '#b0102f' }}>
                No auto-labelled boxes remain (saved boxes count as manual), so every box is scanned. Pick a label above to compare like with like.
            </p>}
            {states.length > 0 && <p className="tp-note" style={{ marginTop: 8 }}>
                {states.map(s => `${s.state} (${s.source}): ${s.count}`).join(' · ')}
            </p>}

            {running && <>
                <div className="tp-progress"><div style={{ width: `${Math.round(100 * (st.done || 0) / Math.max(st.total || 1, 1))}%` }} /></div>
                <div className="tp-note">{st.done || 0} / {st.total || 0} boxes</div>
            </>}

            {sm.total > 0 && <>
                <div className="tp-tiles">
                    <div className="tp-tile"><b>{sm.total}</b><span>scanned</span></div>
                    <div className="tp-tile"><b>{sm.median}</b><span>normal darkness</span></div>
                    <div className="tp-tile"><b>{sm.fence}</b><span>outlier above</span></div>
                    <div className="tp-tile tp-tile--bad"><b>{sm.outliers}</b><span>outliers to check</span></div>
                    <div className="tp-tile tp-tile--good"><b>{reviewed}</b><span>reviewed here</span></div>
                </div>
                <div className="tp-row" style={{ alignItems: 'center' }}>
                    <div className="tp-tabs" style={{ border: 0 }}>
                        <button className={`tp-tab ${outliersOnly ? 'tp-tab--on' : ''}`} onClick={() => { setOutliersOnly(true); setPage(0); }}>Outliers ({sm.outliers})</button>
                        <button className={`tp-tab ${!outliersOnly ? 'tp-tab--on' : ''}`} onClick={() => { setOutliersOnly(false); setPage(0); }}>All, darkest first ({sm.total})</button>
                    </div>
                    <label className="tp-field" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                        <input type="checkbox" checked={enhance} onChange={e => setEnhance(e.target.checked)} style={{ minWidth: 0 }} />
                        Brighten preview
                    </label>
                    <button className="tp-btn tp-btn--ghost" disabled={page === 0} onClick={() => setPage(p => p - 1)}>‹</button>
                    <span className="tp-note">page {page + 1} / {pages}</span>
                    <button className="tp-btn tp-btn--ghost" disabled={page + 1 >= pages} onClick={() => setPage(p => p + 1)}>›</button>
                </div>

                <div className="tp-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))' }}>
                    {pageItems.map(it => {
                        const d = done[it.ann_id];
                        return (
                            <div key={it.ann_id} className={`tp-thumb ${it.outlier ? 'tp-thumb--bad' : ''}`} style={{ cursor: 'default', opacity: d ? 0.55 : 1 }}>
                                <Crop url={`${base}/crop/${it.ann_id}?enhance=${enhance ? 1 : 0}`} />
                                <div>
                                    <span className="tp-tag">{it.state}</span>
                                    dark {it.dark_score} · z {it.z}
                                    <div className="tp-note" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.filename}</div>
                                    {d ? (
                                        <div style={{ marginTop: 4 }}>
                                            <b>{d === 'kept' ? '✓ kept' : `→ ${d}`}</b>{' '}
                                            <button className="tp-btn tp-btn--ghost" style={{ padding: '2px 8px', fontSize: 11 }} onClick={() => undo(it.ann_id)}>Undo</button>
                                        </div>
                                    ) : (
                                        <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                            <button className="tp-btn tp-btn--ghost" style={{ padding: '2px 8px', fontSize: 11 }}
                                                onClick={() => act('confirm', { ann_id: it.ann_id }, 'kept', it.ann_id)}>Keep</button>
                                            {stateNames.filter(s => s !== it.state).map(s => (
                                                <button key={s} className="tp-btn" style={{ padding: '2px 8px', fontSize: 11 }}
                                                    onClick={() => act('relabel', { ann_id: it.ann_id, new_state: s }, s, it.ann_id)}>→ {s}</button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
                {items.length === 0 && <p className="tp-note">No outliers found.</p>}
            </>}
            {st.state === 'done' && !(sm.total > 0) && <p className="tp-err">Nothing matched these filters. Untick "Only auto-labelled" or choose another label.</p>}
        </div>
    );
}
