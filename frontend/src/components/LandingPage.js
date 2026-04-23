import React, { useState, useEffect } from 'react';
import './LandingPage.css';
import { Tag, Zap, Rocket, BarChart2, Target, Brain, FolderOpen, Pencil, ArrowRight, Eye } from 'lucide-react';
import logoImg from '../logo.png';

const LandingPage = ({ onLogin, onGetStarted }) => {
    const [scrolled, setScrolled] = useState(false);
    const [activeFeature, setActiveFeature] = useState(0);

    useEffect(() => {
        const handler = () => setScrolled(window.scrollY > 24);
        window.addEventListener('scroll', handler);
        return () => window.removeEventListener('scroll', handler);
    }, []);

    useEffect(() => {
        const timer = setInterval(() => setActiveFeature(p => (p + 1) % 6), 3000);
        return () => clearInterval(timer);
    }, []);

    const features = [
        {
            icon: <Tag size={24} />,
            title: 'Smart Annotation',
            desc: 'Draw bounding boxes with pixel-perfect precision. Multi-class support with keyboard shortcuts.',
            color: '#dc143c',
        },
        {
            icon: <Zap size={24} />,
            title: 'Auto-Annotate',
            desc: 'Use seed models to auto-label remaining images. Review and accept in one click.',
            color: '#f59e0b',
        },
        {
            icon: <Rocket size={24} />,
            title: 'YOLO Training',
            desc: 'Train YOLO11, v10, v9, v8 models with live epoch charts, loss curves, and mAP metrics.',
            color: '#10b981',
        },
        {
            icon: <BarChart2 size={24} />,
            title: 'Live Metrics',
            desc: 'Watch box loss, cls loss, and mAP50 update in real-time as your model trains.',
            color: '#ec4899',
        },
        {
            icon: <Target size={24} />,
            title: 'Pipeline Control',
            desc: 'Seed model → Auto-annotate → Full training. Automated pipeline with job queuing.',
            color: '#0ea5e9',
        },
        {
            icon: <Brain size={24} />,
            title: '18 YOLO Models',
            desc: 'Choose from Nano to XL across 4 model families. Pick speed or accuracy.',
            color: '#8b5cf6',
        },
    ];

    const steps = [
        {
            num: '01',
            icon: <FolderOpen size={36} />,
            title: 'Upload Your Images',
            desc: 'Drag & drop or batch upload. We handle formats and resolution automatically.',
        },
        {
            num: '02',
            icon: <Pencil size={36} />,
            title: 'Annotate & Auto-Label',
            desc: 'Draw bounding boxes manually, then use AI to auto-label the rest in seconds.',
        },
        {
            num: '03',
            icon: <Target size={36} />,
            title: 'Train & Export',
            desc: 'One click trains your YOLO model. Monitor live metrics, export when ready.',
        },
    ];

    const stats = [
        { num: '18', label: 'YOLO Models' },
        { num: '∞', label: 'Images Supported' },
        { num: '5x', label: 'Faster Labeling' },
        { num: '99%', label: 'GPU Utilization' },
    ];

    return (
        <div className="lnd-page">

            {/* ── Navbar ── */}
            <nav className={`lnd-nav ${scrolled ? 'lnd-nav--scrolled' : ''}`}>
                <div className="lnd-nav-inner">
                    <div className="lnd-brand">
                        <img src={logoImg} alt="Logo" style={{ height: 20, width: 'auto', objectFit: 'contain' }} />
                        <span className="lnd-brand-text">AI Vision Platform</span>
                    </div>
                    <div className="lnd-nav-links">
                        <a href="#features" className="lnd-nav-link">Features</a>
                        <a href="#how" className="lnd-nav-link">How it works</a>
                    </div>
                    <div className="lnd-nav-ctas">
                        <button className="lnd-btn-ghost" onClick={onLogin}>Sign In</button>
                        <button className="lnd-btn-primary" onClick={onGetStarted}>Get Started</button>
                    </div>
                </div>
            </nav>

            {/* ── Hero ── */}
            <section className="lnd-hero">
                <div className="lnd-hero-bg">
                    <div className="lnd-orb lnd-orb--1" />
                    <div className="lnd-orb lnd-orb--2" />
                    <div className="lnd-orb lnd-orb--3" />
                    <div className="lnd-grid" />
                </div>

                <div className="lnd-hero-inner">
                    <div className="lnd-hero-left">
                        <div className="lnd-badge">
                            <span className="lnd-badge-dot" />
                            Powered by YOLO11 · Latest Models
                        </div>

                        <h1 className="lnd-headline">
                            Annotate, Train &amp;<br />
                            <span className="lnd-headline-grad">Deploy Vision</span><br />
                            Models in Minutes
                        </h1>

                        <p className="lnd-subtitle">
                            The all-in-one platform for building, training, and shipping
                            YOLO-powered object detection models — without the ML engineering overhead.
                        </p>

                        <div className="lnd-hero-ctas">
                            <button className="lnd-cta-primary" onClick={onGetStarted}>
                                Sign Up
                                <span className="lnd-cta-arrow"><ArrowRight size={18} /></span>
                            </button>
                        </div>
                    </div>

                    {/* Dashboard Mockup */}
                    <div className="lnd-hero-right">
                        <div className="lnd-mockup">
                            <div className="lnd-mockup-header">
                                <div className="lnd-mockup-dots">
                                    <span /><span /><span />
                                </div>
                                <span className="lnd-mockup-title">AI Vision Platform — Project Alpha</span>
                                <span className="lnd-mockup-nav">← All Projects</span>
                            </div>
                            <div className="lnd-mockup-body">
                                {/* Sidebar */}
                                <div className="lnd-mockup-sidebar">
                                    <div className="lnd-mock-section">
                                        <div className="lnd-mock-label">Pipeline</div>
                                        <div className="lnd-mock-btn lnd-mock-btn--indigo">Seed Model</div>
                                        <div className="lnd-mock-btn lnd-mock-btn--amber">Auto-Annotate</div>
                                        <div className="lnd-mock-btn lnd-mock-btn--green">Train Main</div>
                                        <div className="lnd-mock-btn lnd-mock-btn--teal">Edit Labels</div>
                                    </div>
                                    <div className="lnd-mock-section">
                                        <div className="lnd-mock-label">Images (12)</div>
                                        {['dog_park.jpg', 'street_cam.jpg', 'store_01.jpg'].map((f, i) => (
                                            <div key={f} className={`lnd-mock-image-item ${i === 1 ? 'lnd-mock-image-item--active' : ''}`}>
                                                <span className="lnd-mock-dot" style={{ background: i === 0 ? '#10b981' : '#dc143c' }} />
                                                <span className="lnd-mock-fname">{f}</span>
                                                <span className={`lnd-mock-badge ${i === 0 ? 'lnd-mock-badge--done' : 'lnd-mock-badge--pen'}`}>
                                                    {i === 0 ? 'annotated' : 'pending'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Canvas */}
                                <div className="lnd-mockup-canvas">
                                    <div className="lnd-mock-canvas-header">
                                        <span className="lnd-mock-canvas-name">street_cam.jpg</span>
                                        <span className="lnd-mock-canvas-hint">Click + drag to annotate</span>
                                    </div>
                                    <div className="lnd-mock-canvas-area">
                                        {/* Fake image background */}
                                        <div className="lnd-mock-img">
                                            {/* Bounding boxes */}
                                            <div className="lnd-mock-bbox lnd-mock-bbox--1">
                                                <span className="lnd-mock-bbox-label" style={{ background: '#dc143c' }}>person</span>
                                            </div>
                                            <div className="lnd-mock-bbox lnd-mock-bbox--2">
                                                <span className="lnd-mock-bbox-label" style={{ background: '#f59e0b' }}>car</span>
                                            </div>
                                            <div className="lnd-mock-bbox lnd-mock-bbox--3">
                                                <span className="lnd-mock-bbox-label" style={{ background: '#10b981' }}>bicycle</span>
                                            </div>
                                        </div>
                                    </div>
                                    {/* Annotations list */}
                                    <div className="lnd-mock-anns">
                                        <div className="lnd-mock-ann-label">Annotations (3)</div>
                                        {[
                                            { cls: 'person', color: '#dc143c' },
                                            { cls: 'car', color: '#f59e0b' },
                                            { cls: 'bicycle', color: '#10b981' },
                                        ].map(({ cls, color }) => (
                                            <div key={cls} className="lnd-mock-ann">
                                                <span className="lnd-mock-ann-dot" style={{ background: color }} />
                                                {cls}
                                                <span className="lnd-mock-ann-del">×</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── Stats bar ── */}
            <div className="lnd-stats-bar">
                <div className="lnd-stats-inner">
                    {stats.map((s, i) => (
                        <div key={i} className="lnd-stat">
                            <span className="lnd-stat-num">{s.num}</span>
                            <span className="lnd-stat-label">{s.label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* ── Features Bento ── */}
            <section className="lnd-features" id="features">
                <div className="lnd-section-inner">
                    <div className="lnd-section-badge">Features</div>
                    <h2 className="lnd-section-title">
                        Everything you need to ship<br />
                        <span className="lnd-grad-text">production-ready vision models</span>
                    </h2>
                    <p className="lnd-section-sub">
                        From raw images to a deployed YOLO model — without touching infrastructure.
                    </p>

                    <div className="lnd-bento">
                        {features.map((f, i) => (
                            <div
                                key={i}
                                className={`lnd-bento-card ${activeFeature === i ? 'lnd-bento-card--active' : ''}`}
                                style={{ '--card-color': f.color }}
                                onMouseEnter={() => setActiveFeature(i)}
                            >
                                <span className="lnd-bento-icon" style={{ color: f.color }}>{f.icon}</span>
                                <h3 className="lnd-bento-title">{f.title}</h3>
                                <p className="lnd-bento-desc">{f.desc}</p>
                                <div className="lnd-bento-glow" />
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── How it works ── */}
            <section className="lnd-how" id="how">
                <div className="lnd-section-inner">
                    <div className="lnd-section-badge">How it works</div>
                    <h2 className="lnd-section-title">
                        Three steps to your<br />
                        <span className="lnd-grad-text">first trained model</span>
                    </h2>

                    <div className="lnd-steps">
                        {steps.map((s, i) => (
                            <React.Fragment key={i}>
                                <div className="lnd-step">
                                    <div className="lnd-step-num">{s.num}</div>
                                    <div className="lnd-step-icon">{s.icon}</div>
                                    <h3 className="lnd-step-title">{s.title}</h3>
                                    <p className="lnd-step-desc">{s.desc}</p>
                                </div>
                                {i < steps.length - 1 && (
                                    <div className="lnd-step-arrow"><ArrowRight size={24} /></div>
                                )}
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── CTA section ── */}
            <section className="lnd-cta-section">
                <div className="lnd-cta-inner">
                    <div className="lnd-cta-orb" />
                    <h2 className="lnd-cta-title">
                        Ready to build your<br />
                        <span className="lnd-grad-text">vision model?</span>
                    </h2>
                    <p className="lnd-cta-sub">
                        Start annotating in seconds. Train your first model in minutes.
                    </p>
                    <div className="lnd-cta-btns">
                        <button className="lnd-cta-primary" onClick={onGetStarted}>
                            Sign Up
                            <span className="lnd-cta-arrow"><ArrowRight size={18} /></span>
                        </button>
                    </div>
                </div>
            </section>

            {/* ── Footer ── */}
            <footer className="lnd-footer">
                <div className="lnd-footer-inner">
                    <div className="lnd-brand lnd-brand--footer">
                        <span className="lnd-brand-icon"><Eye size={18} /></span>
                        <span className="lnd-brand-text">AI Vision Platform</span>
                    </div>
                    <p className="lnd-footer-copy">
                        Built for computer vision engineers. Runs locally. Your data, your models.
                    </p>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
