import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Filter, X, Plus, Trash2, Save, Eye } from 'lucide-react';
import { API_URL, BASE_URL } from '../config';
import './TrainingHub.css';

const blank = () => ({ name: '', label: '', counts: {}, allow_other: false, total: null });
const numOrNull = (v) => (v === '' || v == null ? null : Math.max(0, parseInt(v, 10) || 0));

const input = { width: 54, padding: '3px 6px', border: '1px solid #d0d0d0', borderRadius: 4, fontSize: 12 };
const text = { padding: '5px 8px', border: '1px solid #d0d0d0', borderRadius: 4, fontSize: 13 };

/** A rule row per project class: how many boxes of it an image must carry. */
function SegmentEditor({ seg, classes, onChange, onDelete }) {
    const set = (patch) => onChange({ ...seg, ...patch });
    const setRange = (cls, key, val) => {
        const cur = seg.counts[cls] || { min: 0, max: null };
        const next = { ...cur, [key]: numOrNull(val) };
        const counts = { ...seg.counts };
        // empty min + empty max = class not constrained
        if (next.min == null && next.max == null) delete counts[cls];
        else counts[cls] = { min: next.min ?? 0, max: next.max };
        set({ counts });
    };
    const setTotal = (key, val) => {
        const cur = seg.total || { min: null, max: null };
        const next = { ...cur, [key]: numOrNull(val) };
        set({ total: next.min == null && next.max == null ? null : { min: next.min ?? 0, max: next.max } });
    };
    // Same number in both boxes: exactly N.
    const exact = (cls, val) => {
        const n = numOrNull(val);
        const counts = { ...seg.counts };
        if (n == null) delete counts[cls]; else counts[cls] = { min: n, max: n };
        set({ counts });
    };
    return (
        <div style={{ border: '1px solid #e2e2e2', borderRadius: 8, padding: 12, marginBottom: 10, background: '#fafafa' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
                <input style={{ ...text, flex: 1 }} placeholder="Segment name (e.g. Fully locked)" value={seg.name}
                    onChange={e => set({ name: e.target.value })} />
                <input style={{ ...text, width: 140 }} placeholder="Label (e.g. locked)" value={seg.label}
                    onChange={e => set({ label: e.target.value })} />
                <button className="hub-icon-btn" onClick={onDelete} title="Delete segment"><Trash2 size={15} /></button>
            </div>
            <table style={{ fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                    <tr style={{ color: '#888', textAlign: 'left' }}>
                        <th style={{ paddingRight: 14 }}>Class</th><th>Exactly</th>
                        <th style={{ paddingLeft: 14 }}>or min</th><th>max</th>
                    </tr>
                </thead>
                <tbody>
                    {classes.map(c => {
                        const r = seg.counts[c];
                        const isExact = r && r.min === r.max;
                        return (
                            <tr key={c}>
                                <td style={{ paddingRight: 14 }}>{c}</td>
                                <td><input style={input} type="number" min="0" value={isExact ? r.min : ''}
                                    placeholder="any" onChange={e => exact(c, e.target.value)} /></td>
                                <td style={{ paddingLeft: 14 }}><input style={input} type="number" min="0"
                                    value={r && !isExact ? r.min : ''} onChange={e => setRange(c, 'min', e.target.value)} /></td>
                                <td><input style={input} type="number" min="0" value={r && !isExact && r.max != null ? r.max : ''}
                                    onChange={e => setRange(c, 'max', e.target.value)} /></td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
            <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, marginTop: 8, color: '#555' }}>
                <input type="checkbox" checked={seg.allow_other} onChange={e => set({ allow_other: e.target.checked })} />
                Allow other classes on the image (unticked: classes not set above must be absent)
            </label>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, marginTop: 8, color: '#333', fontWeight: 600 }}>
                Total annotations on the image:
                min <input style={input} type="number" min="0" placeholder="any" value={seg.total?.min ?? ''}
                    onChange={e => setTotal('min', e.target.value)} />
                max <input style={input} type="number" min="0" placeholder="any" value={seg.total?.max ?? ''}
                    onChange={e => setTotal('max', e.target.value)} />
                <span style={{ fontWeight: 400, color: '#888' }}>(set both to 3 to drop door + lock images)</span>
            </div>
        </div>
    );
}

const PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#0aa5a5', '#c8a400', '#f032e6'];
const colorOf = (cls, all) => PALETTE[Math.max(0, all.indexOf(cls)) % PALETTE.length];

/** Every image of one segment (or the left-out ones) with its boxes drawn on, to check the selection by eye. */
function SegmentGallery({ project, segments, segment, label, classes, onClose }) {
    const [items, setItems] = useState([]);
    const [total, setTotal] = useState(0);
    const [loading, setLoading] = useState(false);
    const PAGE = 24;

    const load = useCallback(async (offset) => {
        setLoading(true);
        try {
            const { data } = await axios.post(`${API_URL}/segments/${project.id}/images`,
                { segments, segment, offset, limit: PAGE });
            setTotal(data.total);
            setItems(prev => (offset === 0 ? data.images : [...prev, ...data.images]));
        } catch (e) { /* leave the gallery as is */ } finally { setLoading(false); }
    }, [project.id, segments, segment]);

    useEffect(() => { load(0); }, [load]);

    return (
        <div className="hub-overlay" style={{ zIndex: 1100 }} onClick={onClose}>
            <div className="hub-modal" style={{ maxWidth: 1000 }} onClick={e => e.stopPropagation()}>
                <div className="hub-header">
                    <div className="hub-header-left">
                        <div>
                            <h2 className="hub-title">{label}</h2>
                            <p className="hub-subtitle">{total} images · showing {items.length}</p>
                        </div>
                    </div>
                    <div className="hub-header-right"><button className="hub-icon-btn" onClick={onClose}><X size={18} /></button></div>
                </div>
                <div style={{ padding: 14, overflowY: 'auto', maxHeight: '72vh' }}>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, marginBottom: 10 }}>
                        {classes.map(c => (
                            <span key={c} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                <i style={{ width: 10, height: 10, background: colorOf(c, classes), display: 'inline-block' }} />{c}
                            </span>
                        ))}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
                        {items.map(im => (
                            <div key={im.id} style={{ border: '1px solid #ddd', borderRadius: 6, overflow: 'hidden' }}>
                                <div style={{ position: 'relative', lineHeight: 0 }}>
                                    <img alt={im.filename} src={`${BASE_URL}${im.filepath}`} style={{ width: '100%' }} loading="lazy" />
                                    <svg viewBox="0 0 1 1" preserveAspectRatio="none"
                                        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
                                        {im.annotations.filter(a => a.bbox && a.bbox.length === 4).map((a, k) => {
                                            const [xc, yc, w, h] = a.bbox;
                                            return <rect key={k} x={xc - w / 2} y={yc - h / 2} width={w} height={h} fill="none"
                                                stroke={colorOf(a.class_name, classes)} strokeWidth="0.006" />;
                                        })}
                                    </svg>
                                </div>
                                <div style={{ fontSize: 11, padding: '4px 6px', color: '#444' }}>
                                    {Object.entries(im.annotations.reduce((m, a) => ({ ...m, [a.class_name]: (m[a.class_name] || 0) + 1 }), {}))
                                        .map(([c, n]) => `${c}${n > 1 ? ` ×${n}` : ''}`).join(' + ') || 'no annotations'}
                                </div>
                            </div>
                        ))}
                    </div>
                    {items.length < total && (
                        <div style={{ textAlign: 'center', marginTop: 12 }}>
                            <button className="hub-btn" disabled={loading} onClick={() => load(items.length)}>
                                {loading ? 'Loading…' : `Load more (${total - items.length} left)`}
                            </button>
                        </div>
                    )}
                    {!loading && total === 0 && <p style={{ fontSize: 12, color: '#888' }}>No images.</p>}
                </div>
            </div>
        </div>
    );
}

/**
 * Data Segments: define which annotated images belong together by what they
 * contain (e.g. door + 2 locks = "locked"), see the counts, and save. Training
 * panels then offer these segments as a filter and use each label as the class.
 */
export default function DataSegmentsPanel({ project, onClose }) {
    const [segments, setSegments] = useState([]);
    const [classes, setClasses] = useState(project.classes || []);
    const [preview, setPreview] = useState(null);
    const [busy, setBusy] = useState(false);
    const [msg, setMsg] = useState(null);
    const [copyName, setCopyName] = useState('');
    const [gallery, setGallery] = useState(null);   // { segment, label }

    useEffect(() => {
        axios.get(`${API_URL}/segments/${project.id}`).then(r => {
            setSegments(r.data.segments || []);
            if (r.data.classes?.length) setClasses(r.data.classes);
        }).catch(() => setMsg({ err: true, text: 'Could not load segments.' }));
    }, [project.id]);

    const run = useCallback(async (fn) => {
        setBusy(true); setMsg(null);
        try { await fn(); } catch (e) {
            const d = e.response?.data?.detail;
            setMsg({ err: true, text: typeof d === 'string' ? d : 'Request failed.' });
        } finally { setBusy(false); }
    }, []);

    const doPreview = () => run(async () => {
        const { data } = await axios.post(`${API_URL}/segments/${project.id}/preview`, { segments });
        setPreview(data);
    });
    const save = () => run(async () => {
        const { data } = await axios.put(`${API_URL}/segments/${project.id}`, { segments });
        setSegments(data.segments);
        setMsg({ text: 'Saved. Tick segments in any training panel to use them.' });
        const p = await axios.post(`${API_URL}/segments/${project.id}/preview`, { segments: data.segments });
        setPreview(p.data);
    });

    const createCopy = () => run(async () => {
        const { data } = await axios.post(`${API_URL}/segments/${project.id}/create-copy`,
            { segments, name: copyName.trim() || null });
        setMsg({
            err: data.missing_files > 0,
            text: `Created project "${data.name}" with ${data.images} images (${data.left_out} images left out). Find it in your project list.`
                + (data.missing_files ? ` ${data.missing_files} matching image file(s) were not found on disk and were skipped (e.g. ${data.missing_examples.join(', ')}).` : ''),
        });
    });

    const addPreset = (kind) => {
        if (kind === 'reglocks') {
            const pick = (re, d) => classes.find(c => re.test(c)) || d;
            const reg = pick(/max/i, 'max_reg'), l = pick(/lhs/i, 'lhs_lock'), r = pick(/rhs/i, 'rhs_lock');
            setSegments(s => [...s, { ...blank(), name: 'Region + both locks', label: 'region_locks',
                counts: { [reg]: { min: 1, max: 1 }, [l]: { min: 1, max: 1 }, [r]: { min: 1, max: 1 } },
                total: { min: 3, max: 3 } }]);
            return;
        }
        const lock = classes.find(c => /^lock/i.test(c)) || 'lock';
        const unlock = classes.find(c => /^unlock/i.test(c)) || 'unlock';
        const door = classes.find(c => /door/i.test(c)) || 'door';
        const mk = (name, label, d, total) => ({ ...blank(), name, label, counts: d, total });
        const ex = (n) => ({ min: n, max: n });
        const tot = { min: 3, max: 3 };
        const presets = {
            locked: mk('Fully locked', 'locked', { [door]: ex(1), [lock]: ex(2), [unlock]: ex(0) }, tot),
            unlocked: mk('Fully unlocked', 'unlocked', { [door]: ex(1), [lock]: ex(0), [unlock]: ex(2) }, tot),
            partial: mk('Partial', 'partial', { [door]: ex(1), [lock]: ex(1), [unlock]: ex(1) }, tot),
        };
        setSegments(s => [...s, presets[kind]]);
    };

    return (
        <div className="hub-overlay" onClick={onClose}>
            <div className="hub-modal" style={{ maxWidth: 820 }} onClick={e => e.stopPropagation()}>
                <div className="hub-header">
                    <div className="hub-header-left">
                        <span className="hub-header-icon"><Filter size={20} /></span>
                        <div>
                            <h2 className="hub-title">Data Segments</h2>
                            <p className="hub-subtitle">{project.name} — choose images by what is annotated on them</p>
                        </div>
                    </div>
                    <div className="hub-header-right">
                        <button className="hub-icon-btn" onClick={onClose}><X size={18} /></button>
                    </div>
                </div>
                <div style={{ padding: '14px 18px', overflowY: 'auto', maxHeight: '70vh' }}>
                    <p style={{ fontSize: 12, color: '#666', marginTop: 0 }}>
                        An image joins the first segment whose rules it meets. The segment <b>label</b> becomes the
                        image's class for the classifier; detector / segmentation runs just train on the matching images.
                    </p>
                    {segments.map((s, i) => (
                        <SegmentEditor key={i} seg={s} classes={classes}
                            onChange={ns => setSegments(all => all.map((x, k) => (k === i ? ns : x)))}
                            onDelete={() => setSegments(all => all.filter((_, k) => k !== i))} />
                    ))}
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
                        <button className="hub-btn" onClick={() => setSegments(s => [...s, blank()])}><Plus size={13} /> New segment</button>
                        <button className="hub-btn" onClick={() => addPreset('reglocks')}>+ 1 region + LHS + RHS lock</button>
                        <button className="hub-btn" onClick={() => addPreset('locked')}>+ Locked</button>
                        <button className="hub-btn" onClick={() => addPreset('unlocked')}>+ Unlocked</button>
                        <button className="hub-btn" onClick={() => addPreset('partial')}>+ Partial</button>
                    </div>

                    {preview && (
                        <div style={{ borderTop: '1px solid #eee', paddingTop: 12 }}>
                            <p style={{ fontSize: 13, margin: '0 0 8px' }}>
                                <b>{preview.matched}</b> of {preview.annotated_images} annotated images match,{' '}
                                <b>{preview.unmatched}</b> match no segment
                                {preview.overlaps ? <> · <span style={{ color: '#b45309' }}>{preview.overlaps} match several (first wins)</span></> : null}
                            </p>
                            {preview.by_annotation_count && (
                                <p style={{ fontSize: 12, color: '#666', margin: '0 0 8px' }}>
                                    Images by number of annotations: {Object.entries(preview.by_annotation_count).map(([n, c]) => `${n} → ${c}`).join(' · ')}
                                </p>
                            )}
                            {preview.segments.map(s => (
                                <div key={s.name} style={{ marginBottom: 10 }}>
                                    <div style={{ fontSize: 12, fontWeight: 600 }}>{s.name} → {s.label}: {s.images} images{' '}
                                        {s.images > 0 && <button className="hub-btn" style={{ padding: '1px 8px', fontSize: 11 }}
                                            onClick={() => setGallery({ segment: s.name, label: `${s.name} → ${s.label}` })}>View images</button>}
                                    </div>
                                    <div style={{ display: 'flex', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
                                        {(s.samples || []).map(im => (
                                            <img key={im.id} alt={im.filename} title={im.filename}
                                                src={`${BASE_URL}${im.filepath}`}
                                                style={{ height: 56, borderRadius: 4, border: '1px solid #ddd' }} />
                                        ))}
                                    </div>
                                </div>
                            ))}
                            {preview.unmatched > 0 && (
                                <div style={{ fontSize: 12, color: '#666' }}>
                                    Left out: {Object.entries(preview.unmatched_combos).map(([k, v]) => `${k} (${v})`).join(' · ')}{' '}
                                    <button className="hub-btn" style={{ padding: '1px 8px', fontSize: 11 }}
                                        onClick={() => setGallery({ segment: '__unmatched__', label: 'Left out (match no segment)' })}>View images</button>
                                </div>
                            )}
                        </div>
                    )}
                    {msg && <p style={{ fontSize: 12, color: msg.err ? '#c0392b' : '#2e7d32' }}>{msg.text}</p>}
                </div>
                {gallery && <SegmentGallery project={project} segments={segments} classes={classes}
                    segment={gallery.segment} label={gallery.label} onClose={() => setGallery(null)} />}
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '10px 18px', borderTop: '1px solid #eee' }}>
                    <input style={{ ...text, flex: 1, minWidth: 0 }} placeholder="Name for a new project copy (optional)"
                        value={copyName} onChange={e => setCopyName(e.target.value)} />
                    <button className="hub-btn" disabled={busy || !segments.length} onClick={createCopy}
                        title="New project containing only the matching images">Create project copy</button>
                    <button className="hub-btn" disabled={busy} onClick={doPreview}><Eye size={13} /> Preview counts</button>
                    <button className="hub-btn" disabled={busy} onClick={save}><Save size={13} /> Save</button>
                </div>
            </div>
        </div>
    );
}
