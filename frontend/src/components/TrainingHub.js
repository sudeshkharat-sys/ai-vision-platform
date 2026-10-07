import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Rocket, Target, Scissors, Type, Layers, X, RefreshCw } from 'lucide-react';
import { API_URL } from '../config';
import './TrainingHub.css';

const fmtDate = (iso) => (iso ? new Date(iso).toLocaleDateString() : null);

function Chip({ label, ok, detail }) {
    return (
        <div className={`hub-chip ${ok ? 'hub-chip--ok' : ''}`}>
            <span>{ok ? '✅' : '○'} {label}</span>
            <span className="hub-chip-detail">{ok ? detail || 'trained' : 'not trained'}</span>
        </div>
    );
}

const modelDetail = (m) => (m?.exists ? [m.file_size_mb && `${m.file_size_mb} MB`, fmtDate(m.modified_at)].filter(Boolean).join(' · ') : null);

/**
 * One card per training method. Picking a card closes the hub and opens that
 * method's existing panel (onOpen), so adding a method = adding a card here.
 */
export default function TrainingHub({ project, onClose, onOpen }) {
    const [details, setDetails] = useState(null);
    const [ocr, setOcr] = useState(null);
    const [cls, setCls] = useState(null);
    const [loading, setLoading] = useState(true);

    const full = project.project_type === 'combined' || project.project_type === 'ocr';

    const load = useCallback(() => {
        setLoading(true);
        const safe = (p) => p.then(r => r.data).catch(() => null);
        Promise.all([
            safe(axios.get(`${API_URL}/pipeline/model-details/${project.id}`)),
            full ? safe(axios.get(`${API_URL}/ocr/model-status/${project.id}`)) : Promise.resolve(null),
            safe(axios.get(`${API_URL}/crop-cls/model-status/${project.id}`)),
        ]).then(([d, o, c]) => { setDetails(d); setOcr(o); setCls(c); })
            .finally(() => setLoading(false));
    }, [project.id, full]);

    useEffect(() => { load(); }, [load]);

    const go = (kind) => { onClose(); onOpen(kind); };

    const cards = [
        {
            key: 'detect', icon: <Target size={20} />, title: 'Object Detection',
            desc: 'Find objects with boxes. Train a quick seed model first, then the full main model.',
            chips: [
                { label: 'Seed', ok: details?.seed?.exists, detail: modelDetail(details?.seed) },
                { label: 'Main', ok: details?.main?.exists, detail: modelDetail(details?.main) },
            ],
            actions: [
                { label: '1 · Train Seed', onClick: () => go('seed') },
                { label: '2 · Train Main', onClick: () => go('main') },
            ],
        },
        full && {
            key: 'segment', icon: <Scissors size={20} />, title: 'Segmentation',
            desc: 'Outline objects with polygon masks drawn with the Segment tool.',
            chips: [
                { label: 'Seed', ok: details?.seg_seed?.exists, detail: modelDetail(details?.seg_seed) },
                { label: 'Main', ok: details?.seg_main?.exists, detail: modelDetail(details?.seg_main) },
            ],
            actions: [{ label: 'Train Segmentation', onClick: () => go('seg') }],
        },
        full && {
            key: 'ocr', icon: <Type size={20} />, title: 'OCR',
            desc: 'Read characters: per-character CNN, line reader (CRNN) or Tesseract.',
            chips: [
                { label: 'Char CNN', ok: ocr?.has_model },
                { label: 'CRNN', ok: ocr?.crnn?.has_model },
            ],
            actions: [{ label: 'Train OCR', onClick: () => go('ocr') }],
        },
        {
            key: 'classifier', icon: <Layers size={20} />, title: 'Classifier',
            desc: 'Cut out a region (e.g. engine) and classify its state (full / cut / no cover).',
            chips: [
                {
                    label: 'Classifier', ok: cls?.has_classifier,
                    detail: cls?.meta?.val_accuracy != null
                        ? `val acc ${(cls.meta.val_accuracy * 100).toFixed(1)}%` : null,
                },
                { label: 'Region detector', ok: cls?.has_detector },
            ],
            actions: [{ label: 'Train Classifier', onClick: () => go('classifier') }],
        },
    ].filter(Boolean);

    return (
        <div className="hub-overlay" onClick={onClose}>
            <div className="hub-modal" onClick={e => e.stopPropagation()}>
                <div className="hub-header">
                    <div className="hub-header-left">
                        <span className="hub-header-icon"><Rocket size={20} /></span>
                        <div>
                            <h2 className="hub-title">Training Hub</h2>
                            <p className="hub-subtitle">{project.name}</p>
                        </div>
                    </div>
                    <div className="hub-header-right">
                        <button className="hub-icon-btn" onClick={load} title="Refresh status"><RefreshCw size={15} /></button>
                        <button className="hub-icon-btn" onClick={onClose}><X size={18} /></button>
                    </div>
                </div>
                <div className={`hub-grid ${loading ? 'hub-grid--loading' : ''}`}>
                    {cards.map(c => (
                        <div key={c.key} className="hub-card">
                            <div className="hub-card-head">
                                <span className="hub-card-icon">{c.icon}</span>
                                <h3>{c.title}</h3>
                            </div>
                            <p className="hub-card-desc">{c.desc}</p>
                            <div className="hub-chips">
                                {c.chips.map(ch => <Chip key={ch.label} {...ch} />)}
                            </div>
                            <div className="hub-actions">
                                {c.actions.map(a => (
                                    <button key={a.label} className="hub-btn" onClick={a.onClick}>{a.label}</button>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
