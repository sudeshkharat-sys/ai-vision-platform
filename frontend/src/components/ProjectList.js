import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './ProjectList.css';

import { API_URL } from '../config';
import { FolderOpen, Tag, Plus, X, Trash2, AlertTriangle, ArrowRight, Calendar, Grid3X3, Sparkles } from 'lucide-react';
import logoImg from '../logo.png';

/* ── Per-project gradient palette ─────────────────────────────── */
const CARD_PALETTES = [
    { from: '#6366f1', to: '#8b5cf6', light: 'rgba(99,102,241,0.12)',  text: '#a5b4fc' },
    { from: '#06b6d4', to: '#3b82f6', light: 'rgba(6,182,212,0.12)',   text: '#67e8f9' },
    { from: '#10b981', to: '#06b6d4', light: 'rgba(16,185,129,0.12)',  text: '#6ee7b7' },
    { from: '#f59e0b', to: '#ef4444', light: 'rgba(245,158,11,0.12)',  text: '#fcd34d' },
    { from: '#ec4899', to: '#8b5cf6', light: 'rgba(236,72,153,0.12)',  text: '#f9a8d4' },
    { from: '#f97316', to: '#eab308', light: 'rgba(249,115,22,0.12)',  text: '#fdba74' },
];

/* ── Greeting based on time of day ───────────────────────────── */
function getGreeting() {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 18) return 'Good afternoon';
    return 'Good evening';
}

/* ── Skeleton loading card ───────────────────────────────────── */
function SkeletonCard() {
    return (
        <div className="pl-skeleton-card">
            <div className="pl-skeleton-bar" />
            <div className="pl-skeleton-body">
                <div className="pl-skeleton-circle" />
                <div className="pl-skeleton-lines">
                    <div className="pl-skeleton-line pl-skeleton-line--lg" />
                    <div className="pl-skeleton-line pl-skeleton-line--sm" />
                </div>
            </div>
            <div className="pl-skeleton-tags">
                <div className="pl-skeleton-tag" />
                <div className="pl-skeleton-tag" />
                <div className="pl-skeleton-tag" />
            </div>
            <div className="pl-skeleton-footer" />
        </div>
    );
}

/* ── Delete Confirmation Modal ───────────────────────────────── */
function DeleteModal({ project, onConfirm, onCancel, deleting }) {
    const modalRef = useRef(null);

    // Close on Escape key
    useEffect(() => {
        const handler = (e) => { if (e.key === 'Escape' && !deleting) onCancel(); };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [deleting, onCancel]);

    // Close on backdrop click
    const handleBackdrop = (e) => {
        if (!deleting && e.target === e.currentTarget) onCancel();
    };

    return (
        <div className="pl-modal-backdrop" onClick={handleBackdrop}>
            <div className="pl-modal" ref={modalRef} role="dialog" aria-modal="true">
                {/* Icon */}
                <div className="pl-modal-icon">
                    <Trash2 size={28} />
                </div>

                {/* Text */}
                <h2 className="pl-modal-title">Delete Project?</h2>
                <p className="pl-modal-body">
                    You're about to permanently delete{' '}
                    <strong>"{project.name}"</strong>.
                    <br /><br />
                    This will remove <strong>all images, annotations, and training jobs</strong> associated with this project. This action{' '}
                    <strong>cannot be undone</strong>.
                </p>

                {/* Actions */}
                <div className="pl-modal-actions">
                    <button
                        className="pl-modal-cancel"
                        onClick={onCancel}
                        disabled={deleting}
                    >
                        Cancel
                    </button>
                    <button
                        className="pl-modal-delete"
                        onClick={onConfirm}
                        disabled={deleting}
                    >
                        {deleting
                            ? <><span className="pl-btn-spinner" />Deleting…</>
                            : <><Trash2 size={14} /> Delete Project</>
                        }
                    </button>
                </div>
            </div>
        </div>
    );
}

/* ── Project Card ─────────────────────────────────────────────── */
function ProjectCard({ project, index, onClick, onDelete }) {
    const palette = CARD_PALETTES[index % CARD_PALETTES.length];
    const initials = project.name
        .split(' ')
        .slice(0, 2)
        .map(w => w[0]?.toUpperCase() || '')
        .join('');

    const classCount  = project.classes?.length || 0;
    const visClasses  = project.classes?.slice(0, 4) || [];
    const extraCount  = classCount > 4 ? classCount - 4 : 0;

    const handleDeleteClick = (e) => {
        e.stopPropagation();  // Don't open the project
        onDelete(project);
    };

    return (
        <div
            className="pl-card"
            onClick={onClick}
            style={{ '--from': palette.from, '--to': palette.to, '--light': palette.light }}
        >
            {/* Top gradient accent bar */}
            <div className="pl-card-bar" />

            {/* Header row: initials + open hint */}
            <div className="pl-card-head">
                <div className="pl-card-initials" style={{ background: `linear-gradient(135deg, ${palette.from}, ${palette.to})` }}>
                    {initials || '?'}
                </div>
                <div className="pl-card-open-hint">
                    Open <span className="pl-card-arrow"><ArrowRight size={12} /></span>
                </div>
            </div>

            {/* Project name */}
            <div className="pl-card-body">
                <h3 className="pl-card-name">{project.name}</h3>
                {project.description && (
                    <p className="pl-card-desc">{project.description}</p>
                )}
            </div>

            {/* Class tags */}
            <div className="pl-card-tags">
                {classCount === 0 ? (
                    <span className="pl-card-no-classes">No classes yet</span>
                ) : (
                    <>
                        {visClasses.map((cls, i) => (
                            <span
                                key={i}
                                className="pl-card-tag"
                                style={{ '--tag-color': palette.text, '--tag-bg': palette.light }}
                            >
                                <span
                                    className="pl-card-tag-dot"
                                    style={{ background: i % 2 === 0 ? palette.from : palette.to }}
                                />
                                {cls}
                            </span>
                        ))}
                        {extraCount > 0 && (
                            <span className="pl-card-tag pl-card-tag--more">+{extraCount}</span>
                        )}
                    </>
                )}
            </div>

            {/* Footer: stats on left, delete button on right */}
            <div className="pl-card-footer">
                <div className="pl-card-footer-left">
                    <span className="pl-card-stat">
                        <Grid3X3 size={13} />
                        {classCount} {classCount === 1 ? 'class' : 'classes'}
                    </span>
                    <span className="pl-card-stat pl-card-date">
                        <Calendar size={12} />
                        {project.created_at
                            ? new Date(project.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
                            : 'Recently'}
                    </span>
                </div>

                {/* Delete button — in footer, visible on card hover */}
                <button
                    className="pl-card-delete-btn"
                    onClick={handleDeleteClick}
                    title="Delete project"
                    aria-label="Delete project"
                >
                    <Trash2 size={13} />
                </button>
            </div>

            {/* Hover glow overlay */}
            <div className="pl-card-glow" />
        </div>
    );
}

/* ── Main Component ───────────────────────────────────────────── */
const ProjectList = ({ onProjectSelect, user }) => {
    const [projects, setProjects]           = useState([]);
    const [loading, setLoading]             = useState(true);
    const [error, setError]                 = useState(null);
    const [creating, setCreating]           = useState(false);
    const [showForm, setShowForm]           = useState(false);
    const [newName, setNewName]             = useState('');
    const [newClasses, setNewClasses]       = useState('');

    // Delete state
    const [deleteTarget, setDeleteTarget]   = useState(null);  // project object to delete
    const [deleting, setDeleting]           = useState(false);
    const [deletedIds, setDeletedIds]       = useState(new Set()); // for fade-out animation

    useEffect(() => {
        if (!user) return;
        setLoading(true);
        setError(null);
        axios.get(`${API_URL}/projects`)
            .then(res => setProjects(res.data))
            .catch(() => setError('Could not load projects. Make sure the server is running.'))
            .finally(() => setLoading(false));
    }, [user]);

    const createProject = () => {
        if (!newName.trim()) return;
        setCreating(true);
        axios.post(`${API_URL}/projects`, {
            name: newName.trim(),
            classes: newClasses.split(',').map(c => c.trim()).filter(Boolean)
        })
            .then(res => {
                setProjects(prev => [...prev, res.data]);
                setNewName('');
                setNewClasses('');
                setShowForm(false);
            })
            .catch(() => setError('Failed to create project. Please try again.'))
            .finally(() => setCreating(false));
    };

    const handleKeyDown = (e) => { if (e.key === 'Enter') createProject(); };

    /* ── Delete handlers ─────────────────────────────────────── */
    const handleDeleteRequest = (project) => {
        setDeleteTarget(project);
    };

    const handleDeleteConfirm = async () => {
        if (!deleteTarget) return;
        setDeleting(true);
        try {
            await axios.delete(`${API_URL}/projects/${deleteTarget.id}`);
            // Trigger fade-out animation
            setDeletedIds(prev => new Set([...prev, deleteTarget.id]));
            // Remove from list after animation completes
            setTimeout(() => {
                setProjects(prev => prev.filter(p => p.id !== deleteTarget.id));
                setDeletedIds(prev => { const s = new Set(prev); s.delete(deleteTarget.id); return s; });
            }, 350);
            setDeleteTarget(null);
        } catch {
            setError(`Failed to delete "${deleteTarget.name}". Please try again.`);
            setDeleteTarget(null);
        } finally {
            setDeleting(false);
        }
    };

    const handleDeleteCancel = () => {
        if (!deleting) setDeleteTarget(null);
    };

    const firstName = user?.name?.split(' ')[0] || 'there';
    const totalClasses = projects.reduce((s, p) => s + (p.classes?.length || 0), 0);

    return (
        <div className="pl-root">

            {/* ── Delete confirmation modal ────────────────────── */}
            {deleteTarget && (
                <DeleteModal
                    project={deleteTarget}
                    onConfirm={handleDeleteConfirm}
                    onCancel={handleDeleteCancel}
                    deleting={deleting}
                />
            )}

            {/* ── Hero banner ─────────────────────────────────── */}
            <div className="pl-hero">
                <div className="pl-hero-bg" aria-hidden="true">
                    <div className="pl-hero-orb pl-hero-orb--1" />
                    <div className="pl-hero-orb pl-hero-orb--2" />
                    <div className="pl-hero-grid" />
                </div>
                <div className="pl-hero-inner" style={{ position: 'relative' }}>
                    <div className="pl-hero-text">
                        <p className="pl-hero-greeting">{getGreeting()}, {firstName}</p>
                        <h1 className="pl-hero-title">Your Projects</h1>
                        <p className="pl-hero-sub">Manage your computer-vision annotation projects</p>
                    </div>

                    {/* Stats pills */}
                    <div className="pl-stats">
                        <div className="pl-stat-pill">
                            <span className="pl-stat-icon"><FolderOpen size={16} /></span>
                            <div>
                                <span className="pl-stat-num">{projects.length}</span>
                                <span className="pl-stat-label">Projects</span>
                            </div>
                        </div>
                        <div className="pl-stat-pill">
                            <span className="pl-stat-icon"><Tag size={16} /></span>
                            <div>
                                <span className="pl-stat-num">{totalClasses}</span>
                                <span className="pl-stat-label">Total Classes</span>
                            </div>
                        </div>

                        {/* New Project button */}
                        <button
                            className="pl-new-btn"
                            onClick={() => setShowForm(v => !v)}
                        >
                            <span className="pl-new-btn-icon">{showForm ? <X size={18} /> : <Plus size={18} />}</span>
                            {showForm ? 'Cancel' : 'New Project'}
                        </button>
                    </div>

                </div>
            </div>

            {/* ── Body ─────────────────────────────────────────── */}
            <div className="pl-body">

                {/* Error banner */}
                {error && (
                    <div className="pl-error">
                        <AlertTriangle size={16} />
                        {error}
                        <button onClick={() => setError(null)}><X size={16} /></button>
                    </div>
                )}

                {/* Create project form (slide-in) */}
                {showForm && (
                    <div className="pl-create-card">
                        <div className="pl-create-header">
                            <div className="pl-create-icon"><Sparkles size={20} /></div>
                            <div>
                                <h3 className="pl-create-title">Create New Project</h3>
                                <p className="pl-create-sub">Set a name and initial label classes</p>
                            </div>
                        </div>
                        <div className="pl-create-form">
                            <div className="pl-create-field">
                                <label className="pl-create-label">Project Name <span>*</span></label>
                                <input
                                    className="pl-create-input"
                                    value={newName}
                                    onChange={e => setNewName(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="e.g. Vehicle Detection"
                                    autoFocus
                                />
                            </div>
                            <div className="pl-create-field">
                                <label className="pl-create-label">Classes <span className="pl-create-opt">(optional, comma-separated)</span></label>
                                <input
                                    className="pl-create-input"
                                    value={newClasses}
                                    onChange={e => setNewClasses(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="e.g. car, truck, bus, person"
                                />
                            </div>
                            <div className="pl-create-actions">
                                <button
                                    className="pl-create-cancel"
                                    onClick={() => { setShowForm(false); setNewName(''); setNewClasses(''); }}
                                >
                                    Cancel
                                </button>
                                <button
                                    className="pl-create-submit"
                                    onClick={createProject}
                                    disabled={creating || !newName.trim()}
                                >
                                    {creating
                                        ? <><span className="pl-btn-spinner" />Creating…</>
                                        : <><Plus size={16} /> Create Project</>
                                    }
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* ── Project grid ─────────────────────────────── */}
                {loading ? (
                    <div className="pl-grid">
                        {[0, 1, 2].map(i => <SkeletonCard key={i} />)}
                    </div>
                ) : projects.length === 0 ? (
                    <div className="pl-empty">
                        <div className="pl-empty-art">
                            <div className="pl-empty-circle pl-empty-circle--outer" />
                            <div className="pl-empty-circle pl-empty-circle--inner" />
                            <span className="pl-empty-icon"><FolderOpen size={48} /></span>
                        </div>
                        <h3 className="pl-empty-title">No projects yet</h3>
                        <p className="pl-empty-sub">Create your first project to start annotating images</p>
                        <button className="pl-empty-btn" onClick={() => setShowForm(true)}>
                            + Create Your First Project
                        </button>
                    </div>
                ) : (
                    <>
                        <div className="pl-section-label">
                            {projects.length} {projects.length === 1 ? 'Project' : 'Projects'}
                        </div>
                        <div className="pl-grid">
                            {projects.map((p, i) => (
                                <div
                                    key={p.id}
                                    className={`pl-card-wrapper${deletedIds.has(p.id) ? ' pl-card-wrapper--deleting' : ''}`}
                                >
                                    <ProjectCard
                                        project={p}
                                        index={i}
                                        onClick={() => onProjectSelect(p)}
                                        onDelete={handleDeleteRequest}
                                    />
                                </div>
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default ProjectList;
