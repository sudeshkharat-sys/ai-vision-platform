import React, { useState } from 'react';
import axios from 'axios';
import './ActiveLearningPanel.css';
import { Brain, X, RefreshCw, Pencil, AlertTriangle } from 'lucide-react';

import { API_URL } from '../config';

// ════════════════════════════════════════════════════════════
//  OcrActiveLearningPanel
//  Same goal as ActiveLearningPanel (rank pending images by model
//  uncertainty so the most useful ones get labeled first) but scoring the
//  trained character classifier instead of a YOLO detector — a much
//  simpler single-mode flow that runs synchronously (no job queue).
// ════════════════════════════════════════════════════════════
const OcrActiveLearningPanel = ({ project, onClose, onAnnotateImages }) => {
    const [budget, setBudget]   = useState(10);
    const [loading, setLoading] = useState(false);
    const [error, setError]     = useState(null);
    const [result, setResult]   = useState(null);

    const handleScore = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await axios.post(`${API_URL}/ocr/active-learning/suggest/${project.id}`, { budget });
            setResult(res.data);
        } catch (e) {
            setError(e.response?.data?.detail || 'Scoring failed. Train the OCR model first.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="alp-overlay" onClick={onClose}>
            <div className="alp-panel" onClick={e => e.stopPropagation()}>
                <div className="alp-header">
                    <div className="alp-header-left">
                        <span className="alp-header-icon"><Brain size={20} /></span>
                        <div>
                            <h2 className="alp-title">Active Learning</h2>
                            <p className="alp-subtitle">{project.name} — OCR characters</p>
                        </div>
                    </div>
                    <button className="alp-close" onClick={onClose}><X size={18} /></button>
                </div>

                <div className="alp-body">
                    <section className="alp-section">
                        <p className="alp-section-title">Suggest Images to Review</p>
                        <p className="alp-mode-desc" style={{ marginBottom: 14 }}>
                            Ranks pending photos by how uncertain the trained OCR model is about
                            their characters (lowest confidence first, no-detection photos worst)
                            so you label the hardest ones instead of guessing which to do next.
                        </p>
                        <label className="alp-label">
                            How many to suggest
                            <input
                                type="number" min="1" max="200"
                                value={budget}
                                onChange={e => setBudget(Math.max(1, parseInt(e.target.value, 10) || 1))}
                                style={{ width: 80, padding: '6px 8px', border: '1px solid #e5e5e5', borderRadius: 6, fontSize: 13 }}
                            />
                        </label>
                    </section>

                    {error && (
                        <section className="alp-section">
                            <div className="alp-error-banner">
                                <AlertTriangle size={14} /> {error}
                            </div>
                        </section>
                    )}

                    {result && (
                        <section className="alp-section">
                            {result.suggestions.length === 0 ? (
                                <p className="alp-hint">No pending images to score.</p>
                            ) : (
                                <div className="alp-result-table-wrap">
                                    <div className="alp-result-table-header">
                                        <div className="alp-result-table-title">
                                            Most uncertain
                                            <span className="alp-result-count"> ({result.suggestions.length} of {result.total_pending})</span>
                                        </div>
                                        {onAnnotateImages && (
                                            <button
                                                className="alp-annotate-btn"
                                                onClick={() => onAnnotateImages(result.suggestions.map(s => s.image_id))}
                                            >
                                                <Pencil size={12} /> Annotate These
                                            </button>
                                        )}
                                    </div>
                                    <div className="alp-result-table">
                                        {result.suggestions.map((item, i) => (
                                            <div key={item.image_id} className="alp-result-row">
                                                <span className="alp-result-rank">#{i + 1}</span>
                                                <div className="alp-result-info">
                                                    <span className="alp-result-filename">{item.filename}</span>
                                                    <span className="alp-result-reason">{item.reason}</span>
                                                </div>
                                                <span className="alp-result-score">
                                                    {item.char_count > 0 ? `${Math.round(item.avg_confidence * 100)}%` : '—'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </section>
                    )}
                </div>

                <div className="alp-footer">
                    <button className="alp-start-btn" onClick={handleScore} disabled={loading}>
                        {loading ? 'Scoring…' : <><RefreshCw size={16} /> Score Pending Images</>}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default OcrActiveLearningPanel;
