// Centralized API configuration.
// REACT_APP_API_URL is injected at build time via Docker ARG / .env.
// Default: relative URL so Nginx proxies /api/v1 → backend:8000 with no
// hardcoded hostname — works on any domain.
export const API_URL = process.env.REACT_APP_API_URL || "/api/v1";

// Base URL for asset URLs (uploads, static files).
// Empty string = relative URL, which is correct when served through Nginx.
export const BASE_URL = process.env.REACT_APP_BASE_URL || API_URL.replace("/api/v1", "");
