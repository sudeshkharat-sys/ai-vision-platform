import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import './LabelsPanel.css';
import { Tag, X, Check, Pencil, Trash2, ChevronUp, ChevronDown, AlertTriangle, Plus } from 'lucide-react';
import logoImg from '../logo.png';

import { API_URL } from '../config';

// Consistent color per class index (cycles through palette)
const CLASS_COLORS = [
    '#6366f1', '#f59e0b', '#10b981', '#ec4899', '#38bdf8',
    '#f97316', '#a78bfa', '#34d399', '#fb7185', '#facc15',
    '#64748b', '#06b6d4', '#84cc16', '#e879f9', '#f43f5e',
];
const colorFor = (idx) => CLASS_COLORS[idx % CLASS_COLORS.length];

// ══════════════════════════════════════════════════════
//  LabelsPanel
// ══════════════════════════════════════════════════════
const LabelsPanel = ({ project, onClose, onLabelsUpdated }) => {
    const [classes, setClasses]               = useState([...(project.classes || [])]);
    const [classCounts, setClassCounts]       = useState({});
    const [countsLoading, setCountsLoading]   = useState(true);

    // Rename state
    const [editingIdx, setEditingIdx]         = useState(null);
    const [editValue, setEditValue]           = useState('');
    const renameInputRef                       = useRef(null);

    // Delete confirmation state
    const [deletingClass, setDeletingClass]   = useState(null); // { name, count }
    const [deleteAnns, setDeleteAnns]         = useState(false);

    // Add new label state
    const [showAdd, setShowAdd]               = useState(false);
    const [addValue, setAddValue]             = useState('');
    const addInputRef                          = useRef(null);

    const [saving, setSaving]                 = useState(false);
    const [error, setError]                   = useState(null);

    // ── Load annotation counts per class ─────────────────────────
    const loadCounts = useCallback(() => {
        setCountsLoading(true);
        axios.get(`${API_URL}/projects/${project.id}/class-stats`)
            .then(res => setClassCounts(res.data))
            .catch(() => {})
            .finally(() => setCountsLoading(false));
    }, [project.id]);

    useEffect(() => { loadCounts(); }, [loadCounts]);

    // Focus rename input when editing starts
    useEffect(() => {
        if (editingIdx !== null && renameInputRef.current) {
            renameInputRef.current.focus();
            renameInputRef.current.select();
        }
    }, [editingIdx]);

    // Focus add input when add form opens
    useEffect(() => {
        if (showAdd && addInputRef.current) addInputRef.current.focus();
    }, [showAdd]);

    const totalAnnotations = Object.values(classCounts).reduce((a, b) => a + b, 0);

    // ── Add new label ─────────────────────────────────────────────
    const handleAdd = async () => {
        const name = addValue.trim();
        if (!name) return;
        if (classes.map(c => c.toLowerCase()).includes(name.toLowerCase())) {
            setError(`"${name}" already exists.`);
            return;
        }
        setSaving(true);
        setError(null);
        try {
            const res = await axios.patch(`${API_URL}/projects/${project.id}`, {
                classes: [...classes, name],
            });
            const updated = res.data.classes;
            setClasses(updated);
            setAddValue('');
            setShowAdd(false);
            onLabelsUpdated(updated);
        } catch {
            setError('Failed to add label.');
        } finally {
            setSaving(false);
        }
    };

    // ── Rename ────────────────────────────────────────────────────
    const startEdit = (idx) => {
        setEditingIdx(idx);
        setEditValue(classes[idx]);
        setError(null);
        setDeletingClass(null);
    };

    const cancelEdit = () => {
        setEditingIdx(null);
        setEditValue('');
        setError(null);
    };

    const saveEdit = async (idx) => {
        const newName = editValue.trim();
        const oldName = classes[idx];
        if (!newName) { cancelEdit(); return; }
        if (newName === oldName) { cancelEdit(); return; }
        if (classes.some((c, i) => i !== idx && c.toLowerCase() === newName.toLowerCase())) {
            setError(`"${newName}" already exists.`);
            return;
        }
        setSaving(true);
        setError(null);
        try {
            const res = await axios.patch(
                `${API_URL}/projects/${project.id}/rename-class`,
                { old_name: oldName, new_name: newName },
            );
            const updated = res.data.updated_classes;
            setClasses(updated);
            setEditingIdx(null);
            // Update count map key
            setClassCounts(prev => {
                const next = { ...prev };
                if (next[oldName] !== undefined) {
                    next[newName] = next[oldName];
                    delete next[oldName];
                }
                return next;
            });
            onLabelsUpdated(updated);
        } catch {
            setError('Failed to rename label.');
        } finally {
            setSaving(false);
        }
    };

    // ── Delete ────────────────────────────────────────────────────
    const confirmDelete = (name) => {
        setDeletingClass({ name, count: classCounts[name] || 0 });
        setDeleteAnns(false);
        setError(null);
        setEditingIdx(null);
    };

    const cancelDelete = () => {
        setDeletingClass(null);
        setDeleteAnns(false);
    };

    const executeDelete = async () => {
        if (!deletingClass) return;
        setSaving(true);
        setError(null);
        try {
            if (deleteAnns && deletingClass.count > 0) {
                await axios.delete(
                    `${API_URL}/projects/${project.id}/class-annotations/${encodeURIComponent(deletingClass.name)}`
                );
            }
            const newClasses = classes.filter(c => c !== deletingClass.name);
            const res = await axios.patch(`${API_URL}/projects/${project.id}`, {
                classes: newClasses,
            });
            const updated = res.data.classes;
            setClasses(updated);
            setClassCounts(prev => {
                const next = { ...prev };
                delete next[deletingClass.name];
                return next;
            });
            setDeletingClass(null);
            onLabelsUpdated(updated);
        } catch {
            setError('Failed to delete label.');
        } finally {
            setSaving(false);
        }
    };

    // ── Reorder ───────────────────────────────────────────────────
    const move = async (idx, dir) => {
        const target = idx + dir;
        if (target < 0 || target >= classes.length || saving) return;
        const newClasses = [...classes];
        [newClasses[idx], newClasses[target]] = [newClasses[target], newClasses[idx]];
        setSaving(true);
        setError(null);
        try {
            const res = await axios.patch(`${API_URL}/projects/${project.id}`, {
                classes: newClasses,
            });
            setClasses(res.data.classes);
            onLabelsUpdated(res.data.classes);
        } catch {
            setError('Failed to reorder.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="lp-overlay" onClick={onClose}>
            <div className="lp-panel" onClick={e => e.stopPropagation()}>

                {/* ── Header ── */}
                <div className="lp-header">
                    <div className="lp-header-left">
                        <span className="lp-header-icon"><Tag size={20} /></span>
                        <div>
                            <h2 className="lp-title">Edit Labels</h2>
                            <p className="lp-subtitle">{project.name}</p>
                        </div>
                    </div>
                    <button className="lp-close" onClick={onClose}><X size={18} /></button>
                </div>

                <div className="lp-body">

                    {/* Stats row */}
                    <div className="lp-stats-row">
                        <div className="lp-stat-chip">
                            <span className="lp-stat-num">{classes.length}</span>
                            <span className="lp-stat-label">labels</span>
                        </div>
                        <div className="lp-stat-chip">
                            <span className="lp-stat-num">
                                {countsLoading ? '…' : totalAnnotations}
                            </span>
                            <span className="lp-stat-label">annotations</span>
                        </div>
                    </div>

                    {/* Error banner */}
                    {error && (
                        <div className="lp-error">
                            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><AlertTriangle size={14} /> {error}</span>
                            <button className="lp-error-dismiss" onClick={() => setError(null)}><X size={14} /></button>
                        </div>
                    )}

                    {/* Delete confirmation card */}
                    {deletingClass && (
                        <div className="lp-confirm-card">
                            <p className="lp-confirm-title">
                                Delete <strong>"{deletingClass.name}"</strong>?
                            </p>
                            {deletingClass.count > 0 ? (
                                <label className="lp-confirm-check">
                                    <input
                                        type="checkbox"
                                        checked={deleteAnns}
                                        onChange={e => setDeleteAnns(e.target.checked)}
                                    />
                                    <span>
                                        Also delete <strong>{deletingClass.count}</strong> annotation{deletingClass.count !== 1 ? 's' : ''} with this class
                                    </span>
                                </label>
                            ) : (
                                <p className="lp-confirm-sub">No annotations use this class.</p>
                            )}
                            <div className="lp-confirm-btns">
                                <button
                                    className="lp-btn lp-btn--ghost"
                                    onClick={cancelDelete}
                                    disabled={saving}
                                >
                                    Cancel
                                </button>
                                <button
                                    className="lp-btn lp-btn--danger"
                                    onClick={executeDelete}
                                    disabled={saving}
                                >
                                    {saving ? 'Deleting…' : 'Delete'}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Label list */}
                    <div className="lp-list">
                        {classes.length === 0 ? (
                            <div className="lp-empty">
                                <span className="lp-empty-icon"><Tag size={32} /></span>
                                <p>No labels yet.</p>
                                <p className="lp-empty-sub">Add labels below to organize your annotations.</p>
                            </div>
                        ) : (
                            classes.map((cls, idx) => {
                                const isEditing   = editingIdx === idx;
                                const isDeleting  = deletingClass?.name === cls;
                                const count       = classCounts[cls] ?? 0;

                                return (
                                    <div
                                        key={`${cls}-${idx}`}
                                        className={[
                                            'lp-row',
                                            isEditing  ? 'lp-row--editing'  : '',
                                            isDeleting ? 'lp-row--deleting' : '',
                                        ].join(' ')}
                                    >
                                        {/* Color dot */}
                                        <span
                                            className="lp-dot"
                                            style={{ background: colorFor(idx) }}
                                        />

                                        {/* Name / rename input */}
                                        {isEditing ? (
                                            <input
                                                ref={renameInputRef}
                                                className="lp-rename-input"
                                                value={editValue}
                                                onChange={e => setEditValue(e.target.value)}
                                                onKeyDown={e => {
                                                    if (e.key === 'Enter') saveEdit(idx);
                                                    if (e.key === 'Escape') cancelEdit();
                                                }}
                                            />
                                        ) : (
                                            <span className="lp-name">{cls}</span>
                                        )}

                                        {/* Annotation count */}
                                        <span className="lp-count" title={`${count} annotation${count !== 1 ? 's' : ''}`}>
                                            {countsLoading ? '…' : count}
                                        </span>

                                        {/* Action buttons */}
                                        {isEditing ? (
                                            <div className="lp-row-actions">
                                                <button
                                                    className="lp-icon-btn lp-icon-btn--save"
                                                    onClick={() => saveEdit(idx)}
                                                    disabled={saving}
                                                    title="Save rename"
                                                ><Check size={14} /></button>
                                                <button
                                                    className="lp-icon-btn lp-icon-btn--cancel"
                                                    onClick={cancelEdit}
                                                    title="Cancel"
                                                ><X size={14} /></button>
                                            </div>
                                        ) : (
                                            <div className="lp-row-actions">
                                                <button
                                                    className="lp-icon-btn"
                                                    onClick={() => move(idx, -1)}
                                                    disabled={idx === 0 || saving}
                                                    title="Move up"
                                                ><ChevronUp size={14} /></button>
                                                <button
                                                    className="lp-icon-btn"
                                                    onClick={() => move(idx, 1)}
                                                    disabled={idx === classes.length - 1 || saving}
                                                    title="Move down"
                                                ><ChevronDown size={14} /></button>
                                                <button
                                                    className="lp-icon-btn lp-icon-btn--edit"
                                                    onClick={() => startEdit(idx)}
                                                    disabled={saving}
                                                    title="Rename"
                                                ><Pencil size={14} /></button>
                                                <button
                                                    className="lp-icon-btn lp-icon-btn--del"
                                                    onClick={() => confirmDelete(cls)}
                                                    disabled={saving}
                                                    title="Delete"
                                                ><Trash2 size={14} /></button>
                                            </div>
                                        )}
                                    </div>
                                );
                            })
                        )}
                    </div>

                    {/* Add new label */}
                    {showAdd ? (
                        <div className="lp-add-row">
                            <input
                                ref={addInputRef}
                                className="lp-add-input"
                                placeholder="Label name…"
                                value={addValue}
                                onChange={e => setAddValue(e.target.value)}
                                onKeyDown={e => {
                                    if (e.key === 'Enter') handleAdd();
                                    if (e.key === 'Escape') { setShowAdd(false); setAddValue(''); setError(null); }
                                }}
                            />
                            <button
                                className="lp-btn lp-btn--primary"
                                onClick={handleAdd}
                                disabled={!addValue.trim() || saving}
                            >
                                {saving ? 'Adding…' : 'Add'}
                            </button>
                            <button
                                className="lp-btn lp-btn--ghost"
                                onClick={() => { setShowAdd(false); setAddValue(''); setError(null); }}
                                disabled={saving}
                            >
                                Cancel
                            </button>
                        </div>
                    ) : (
                        <button className="lp-add-btn" onClick={() => { setShowAdd(true); setEditingIdx(null); }}>
                            <Plus size={16} /> Add Label
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LabelsPanel;
