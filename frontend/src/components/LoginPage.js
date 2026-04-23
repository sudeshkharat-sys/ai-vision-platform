import React, { useState } from 'react';
import axios from 'axios';
import './LoginPage.css';

import { API_URL } from '../config';
import { Eye, EyeOff, ArrowRight, Zap, Target, Pencil, Rocket, AlertTriangle } from 'lucide-react';
import logoImg from '../logo.png';

// ── Left panel feature list ──────────────────────────────────────
const FEATURES = [
    { icon: <Pencil size={16} />, title: 'Precise Annotation', desc: 'Draw bounding boxes on any image with pixel-perfect control.' },
    { icon: <Zap size={16} />, title: 'Auto-Label with AI', desc: 'Let a seed YOLO model annotate hundreds of images in seconds.' },
    { icon: <Rocket size={16} />, title: 'One-Click Training', desc: 'Train YOLO11 · v10 · v9 · v8 with live loss charts & metrics.' },
    { icon: <Target size={16} />, title: 'Full Pipeline', desc: 'Seed → Auto-Annotate → Main Model. Automated job queuing.' },
];

// ── Left Panel ───────────────────────────────────────────────────
const LeftPanel = () => (
    <div className="lp2-left">
        {/* Background layers */}
        <div className="lp2-left-bg">
            <div className="lp2-orb lp2-orb--1" />
            <div className="lp2-orb lp2-orb--2" />
            <div className="lp2-orb lp2-orb--3" />
            <div className="lp2-grid" />
        </div>

        <div className="lp2-left-content">
            {/* Brand */}
            <div className="lp2-brand">
                <span className="lp2-brand-icon"><Eye size={22} /></span>
                <span className="lp2-brand-name">AI Vision Platform</span>
            </div>

            {/* Headline */}
            <div className="lp2-headline-wrap">
                <h2 className="lp2-headline">
                    Annotate. Train.<br />
                    <span className="lp2-headline-grad">Ship faster.</span>
                </h2>
                <p className="lp2-headline-sub">
                    The all-in-one workspace for building YOLO-powered object detection models — without the MLOps headache.
                </p>
            </div>

            {/* Annotation mockup card */}
            <div className="lp2-mockup-card">
                <div className="lp2-mockup-bar">
                    <div className="lp2-mockup-dots">
                        <span /><span /><span />
                    </div>
                    <span className="lp2-mockup-title">street_cam_042.jpg</span>
                </div>
                <div className="lp2-mockup-canvas">
                    {/* Fake image with bounding boxes */}
                    <div className="lp2-fake-img">
                        <div className="lp2-bbox lp2-bbox--person">
                            <span className="lp2-bbox-tag" style={{ background: '#dc143c' }}>person</span>
                        </div>
                        <div className="lp2-bbox lp2-bbox--car">
                            <span className="lp2-bbox-tag" style={{ background: '#f59e0b' }}>car</span>
                        </div>
                        <div className="lp2-bbox lp2-bbox--bike">
                            <span className="lp2-bbox-tag" style={{ background: '#10b981' }}>bicycle</span>
                        </div>
                    </div>
                </div>
                <div className="lp2-mockup-footer">
                    <span className="lp2-tag" style={{ borderColor: '#dc143c80' }}>
                        <span className="lp2-tag-dot" style={{ background: '#dc143c' }} />person
                    </span>
                    <span className="lp2-tag" style={{ borderColor: '#f59e0b80' }}>
                        <span className="lp2-tag-dot" style={{ background: '#f59e0b' }} />car
                    </span>
                    <span className="lp2-tag" style={{ borderColor: '#10b98180' }}>
                        <span className="lp2-tag-dot" style={{ background: '#10b981' }} />bicycle
                    </span>
                </div>
            </div>

            {/* Features */}
            <ul className="lp2-features">
                {FEATURES.map((f, i) => (
                    <li key={i} className="lp2-feature">
                        <span className="lp2-feature-icon">{f.icon}</span>
                        <div>
                            <span className="lp2-feature-title">{f.title}</span>
                            <span className="lp2-feature-desc"> — {f.desc}</span>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    </div>
);

// ── LoginPage ────────────────────────────────────────────────────
const LoginPage = ({ defaultTab = 'login', onSuccess, onBack }) => {
    const [tab, setTab]           = useState(defaultTab);
    const [email, setEmail]       = useState('');
    const [password, setPassword] = useState('');
    const [name, setName]         = useState('');
    const [error, setError]       = useState(null);
    const [loading, setLoading]   = useState(false);
    const [showPass, setShowPass] = useState(false);

    const reset = () => { setError(null); setEmail(''); setPassword(''); setName(''); };
    const switchTab = (t) => { setTab(t); reset(); };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        if (!email.trim() || !password.trim()) { setError('Email and password are required.'); return; }
        if (tab === 'register' && !name.trim()) { setError('Name is required.'); return; }
        if (password.length < 6) { setError('Password must be at least 6 characters.'); return; }

        setLoading(true);
        try {
            const endpoint = tab === 'login' ? '/auth/login' : '/auth/register';
            const payload  = tab === 'login'
                ? { email, password }
                : { email, password, name };
            const res = await axios.post(`${API_URL}${endpoint}`, payload);
            const { token, user } = res.data;
            localStorage.setItem('auth_token', token);
            localStorage.setItem('auth_user', JSON.stringify(user));
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            onSuccess(user);
        } catch (err) {
            setError(err.response?.data?.detail || 'Something went wrong. Try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="lp2-page">
            {/* ── Left panel ── */}
            <LeftPanel />

            {/* ── Right panel (form) ── */}
            <div className="lp2-right">

                {/* Back */}
                <button className="lp2-back" onClick={onBack}>
                    ← Home
                </button>

                {/* Logo top-right */}
                <div className="lp2-logo">
                    <img src={logoImg} alt="Logo" style={{ height: 36, width: 'auto', objectFit: 'contain' }} />
                </div>

                <div className="lp2-form-wrap">
                    {/* Tabs */}
                    <div className="lp2-tabs">
                        <button
                            className={`lp2-tab ${tab === 'login' ? 'lp2-tab--active' : ''}`}
                            onClick={() => switchTab('login')}
                        >
                            Sign In
                        </button>
                        <button
                            className={`lp2-tab ${tab === 'register' ? 'lp2-tab--active' : ''}`}
                            onClick={() => switchTab('register')}
                        >
                            Create Account
                        </button>
                    </div>

                    {/* Heading */}
                    <div className="lp2-form-heading">
                        <h1>
                            {tab === 'login' ? 'Welcome back' : 'Get started free'}
                        </h1>
                        <p>
                            {tab === 'login'
                                ? 'Sign in to your workspace to continue.'
                                : 'Create your account and start annotating in seconds.'}
                        </p>
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="lp2-error">
                            <AlertTriangle size={15} />
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Form */}
                    <form className="lp2-form" onSubmit={handleSubmit} noValidate>
                        {tab === 'register' && (
                            <div className="lp2-field">
                                <label className="lp2-label">Full Name</label>
                                <input
                                    className="lp2-input"
                                    type="text"
                                    placeholder="Jane Smith"
                                    value={name}
                                    onChange={e => setName(e.target.value)}
                                    autoFocus
                                    disabled={loading}
                                />
                            </div>
                        )}

                        <div className="lp2-field">
                            <label className="lp2-label">Email address</label>
                            <input
                                className="lp2-input"
                                type="email"
                                placeholder="you@example.com"
                                value={email}
                                onChange={e => setEmail(e.target.value)}
                                autoFocus={tab === 'login'}
                                disabled={loading}
                            />
                        </div>

                        <div className="lp2-field">
                            <label className="lp2-label">Password</label>
                            <div className="lp2-pass-wrap">
                                <input
                                    className="lp2-input"
                                    type={showPass ? 'text' : 'password'}
                                    placeholder={tab === 'register' ? 'Min. 6 characters' : '••••••••'}
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    disabled={loading}
                                    style={{ paddingRight: 44 }}
                                />
                                <button
                                    type="button"
                                    className="lp2-eye"
                                    onClick={() => setShowPass(p => !p)}
                                    tabIndex={-1}
                                    title={showPass ? 'Hide' : 'Show'}
                                >
                                    {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
                                </button>
                            </div>
                        </div>

                        <button
                            type="submit"
                            className="lp2-submit"
                            disabled={loading}
                        >
                            {loading
                                ? (tab === 'login' ? 'Signing in…' : 'Creating account…')
                                : (tab === 'login' ? 'Sign In' : 'Create Account')}
                        </button>
                    </form>

                    <p className="lp2-switch">
                        {tab === 'login' ? "Don't have an account? " : 'Already have an account? '}
                        <button
                            type="button"
                            className="lp2-switch-btn"
                            onClick={() => switchTab(tab === 'login' ? 'register' : 'login')}
                        >
                            {tab === 'login' ? 'Sign up free' : 'Sign in'}
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
