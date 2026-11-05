Local launchers for ai-vision-platform (Windows, no PowerShell needed).

First time only:
  1. conda create -n ai-vision python=3.11 -y --override-channels -c conda-forge
  2. Copy .env.example to .env (repo root) and fill in the Postgres values.
  3. install-requirements.bat

Daily:
  start-all.bat          (Postgres, Redis, backend, Celery, frontend)
or one by one: start-postgres, start-redis, start-backend, start-celery, start-frontend.
Other: stop-postgres.bat, psql-pg.bat

Paths live in _config.bat (PROJ must match where the repo is on your PC). Edit it if folders move.
Backend runs on port 8000 (Celery uses queue "celery"). Redis must be version 6 or newer.
Postgres and Redis are shared with other projects (e.g. flutter-ai): run one stack at a time,
or use different BACKEND_PORT / FRONTEND_PORT.

If double-clicking a .bat only shows "Open with" (or opens it in VS Code):
  Windows has no default app for .bat on that PC. Do not pick VS Code - it only edits the file.
  Use a Desktop shortcut instead: right-click Desktop > New > Shortcut, Location:
    C:\Windows\System32\cmd.exe /k D:\Codes\Dataeaze\ai-vision-platform\local_dev\start-all.bat
  Tips: type the quotes by hand (curly quotes copied from chat/Word break it), keep the repo path
  free of spaces, leave "Run as administrator" off, and update the shortcut if the folder moves.

Typed shortcuts: run install-shortcuts.bat once, add %USERPROFILE%\dev-tools to your user PATH.
Then in any Anaconda Prompt: vi-all, vi-backend, vi-celery, vi-frontend, pg-start, pg-stop, pg-psql, redis-start.
