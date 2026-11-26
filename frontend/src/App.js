import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Eye, ArrowLeft, LogOut } from 'lucide-react';
import './App.css';
import LandingPage from './components/LandingPage';
import LoginPage from './components/LoginPage';
import ProjectList from './components/ProjectList';
import AnnotationWorkspace from './components/AnnotationWorkspace';
import logoImg from './logo.png';

import { API_URL } from './config';

// ── Attach saved token to every axios request automatically ──────
const savedToken = localStorage.getItem('auth_token');
if (savedToken) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${savedToken}`;
}

// ── Auto-logout on 401 from any request ──────────────────────────
// Compare the token used in the failing request against the current
// token so stale in-flight requests from a previous session never
// accidentally kick the freshly-logged-in user back to the landing page.
axios.interceptors.response.use(
    res => res,
    err => {
        if (err.response?.status === 401) {
            const requestToken = err.config?.headers?.['Authorization'];
            const currentToken = localStorage.getItem('auth_token');
            const currentHeader = currentToken ? `Bearer ${currentToken}` : null;
            if (!currentHeader || requestToken === currentHeader) {
                localStorage.removeItem('auth_token');
                localStorage.removeItem('auth_user');
                delete axios.defaults.headers.common['Authorization'];
                window.dispatchEvent(new Event('auth:logout'));
            }
        }
        return Promise.reject(err);
    }
);

function App() {
    const [view, setView]                 = useState('landing');
    const [loginDefaultTab, setLoginDefaultTab] = useState('login');
    const [currentUser, setCurrentUser]   = useState(null);
    const [currentProject, setCurrentProject] = useState(null);
    const [authChecking, setAuthChecking] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem('auth_token');
        if (!token) { setAuthChecking(false); return; }

        // Ask the server who we are. If the backend is down or restarting there is no
        // answer at all - that must NOT log the user out, so we wait and try again.
        // Only a real 401 answer from the server means the token is bad.
        let tries = 0;
        const checkLogin = () => {
            axios.get(`${API_URL}/auth/me`)
                .then(async res => {
                    setCurrentUser(res.data);
                    setView('dashboard');
                    const savedId = sessionStorage.getItem('current_project_id');
                    if (savedId) {
                        try {
                            const projRes = await axios.get(`${API_URL}/projects/${savedId}`);
                            setCurrentProject(projRes.data);
                        } catch {
                            sessionStorage.removeItem('current_project_id');
                        }
                    }
                    setAuthChecking(false);
                })
                .catch(err => {
                    if (err.response?.status === 401) {
                        // server says the token is invalid -> really log out
                        localStorage.removeItem('auth_token');
                        localStorage.removeItem('auth_user');
                        delete axios.defaults.headers.common['Authorization'];
                        setAuthChecking(false);
                    } else if (tries < 20) {
                        // backend not reachable (restarting?) -> keep the token, retry in 3s
                        tries += 1;
                        setTimeout(checkLogin, 3000);
                    } else {
                        // gave up waiting (about 1 minute); token is kept for the next refresh
                        setAuthChecking(false);
                    }
                });
        };
        checkLogin();
    }, []);

    useEffect(() => {
        const handler = () => {
            sessionStorage.removeItem('current_project_id');
            setCurrentUser(null);
            setCurrentProject(null);
            setView('landing');
        };
        window.addEventListener('auth:logout', handler);
        return () => window.removeEventListener('auth:logout', handler);
    }, []);

    const handleLoginSuccess = (user) => {
        setCurrentUser(user);
        setCurrentProject(null);
        setView('dashboard');
    };

    const handleLogout = () => {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
        sessionStorage.removeItem('current_project_id');
        delete axios.defaults.headers.common['Authorization'];
        setCurrentUser(null);
        setCurrentProject(null);
        setView('landing');
    };

    const handleProjectSelect = (project) => {
        sessionStorage.setItem('current_project_id', project.id);
        setCurrentProject(project);
    };

    const handleBackToDashboard = () => {
        sessionStorage.removeItem('current_project_id');
        setCurrentProject(null);
    };

    const handleProjectUpdated = (updated) => {
        sessionStorage.setItem('current_project_id', updated.id);
        setCurrentProject(updated);
    };

    if (authChecking) {
        return (
            <div className="app-boot">
                <span className="app-boot-icon">
                    <Eye size={40} />
                </span>
            </div>
        );
    }

    if (view === 'landing') {
        return (
            <LandingPage
                onLogin={() => { setLoginDefaultTab('login'); setView('login'); }}
                onGetStarted={() => { setLoginDefaultTab('register'); setView('login'); }}
            />
        );
    }

    if (view === 'login') {
        return (
            <LoginPage
                defaultTab={loginDefaultTab}
                onSuccess={handleLoginSuccess}
                onBack={() => setView('landing')}
            />
        );
    }

    return (
        <div className="app">
            <header className="app-header">
                <div className="app-header-inner">
                    <div className="app-logo">
                        <img src={logoImg} alt="Logo" className="app-logo-img" />
                        <span className="app-logo-text">AI Vision Platform</span>
                    </div>

                    {currentProject && (
                        <nav className="app-nav">
                            <button
                                className="btn-back"
                                onClick={handleBackToDashboard}
                            >
                                <ArrowLeft size={14} />
                                Dashboard
                            </button>
                            <span className="app-nav-project">{currentProject.name}</span>
                        </nav>
                    )}

                    <div className="app-user">
                        <span className="app-user-avatar">
                            {currentUser?.name?.[0]?.toUpperCase() || '?'}
                        </span>
                        <span className="app-user-name">{currentUser?.name}</span>
                        <button className="app-logout-btn" onClick={handleLogout} title="Sign out">
                            <LogOut size={13} style={{ marginRight: 4, verticalAlign: 'middle' }} />
                            Sign Out
                        </button>
                    </div>
                </div>
            </header>

            <main className="app-main">
                {currentProject ? (
                    <AnnotationWorkspace
                        project={currentProject}
                        onProjectUpdated={handleProjectUpdated}
                    />
                ) : (
                    <ProjectList onProjectSelect={handleProjectSelect} user={currentUser} />
                )}
            </main>
        </div>
    );
}

export default App;
