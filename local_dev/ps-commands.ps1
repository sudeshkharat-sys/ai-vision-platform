# ai-vision-platform PowerShell commands. Load from your profile with ONE line:
#   . "D:\Codes\Dataeaze\ai-vision-platform\local_dev\ps-commands.ps1"
# Needs PowerShell scripts to be allowed. If your profile loads, this loads too.
$global:VILocal = $PSScriptRoot

function Start-VIPostgres   { & "$global:VILocal\start-postgres.bat" }
function Stop-VIPostgres    { & "$global:VILocal\stop-postgres.bat" }
function Connect-VIPostgres { & "$global:VILocal\psql-pg.bat" }

function Invoke-VIWindow([string]$Title, [string]$Bat) {
    Start-Process cmd.exe -ArgumentList '/k', "title $Title && `"$global:VILocal\$Bat.bat`""
}
function Start-VIRedis    { Invoke-VIWindow 'VI Redis'    'start-redis' }
function Start-VIBackend  { Invoke-VIWindow 'VI Backend'  'start-backend' }
function Start-VICelery   { Invoke-VIWindow 'VI Celery'   'start-celery' }
function Start-VIFrontend { Invoke-VIWindow 'VI Frontend' 'start-frontend' }

function Start-VIAll {
    Start-VIPostgres
    Start-VIRedis
    Start-Sleep -Seconds 3
    Start-VIBackend
    Start-VICelery
    Start-VIFrontend
}

# ---- Stop / restart one service (closes its window by title) -------------------
function Stop-VIWindow([string]$Title) { taskkill /FI "WINDOWTITLE eq $Title*" /T /F 2>$null | Out-Null }
function Stop-VIBackend  { Stop-VIWindow 'VI Backend' }
function Stop-VICelery   { Stop-VIWindow 'VI Celery' }
function Stop-VIFrontend { Stop-VIWindow 'VI Frontend' }
function Restart-VIBackend  { Stop-VIBackend;  Start-Sleep 1; Start-VIBackend }
function Restart-VICelery   { Stop-VICelery;   Start-Sleep 1; Start-VICelery }
function Restart-VIFrontend { Stop-VIFrontend; Start-Sleep 1; Start-VIFrontend }
function Show-VILog { Get-Content "$global:VILocal\..ackend.log" -Tail 60 }   # last lines of backend.log

# ---- Short names (same as the old typed commands) --------------------------------
Set-Alias vi-all       Start-VIAll
Set-Alias vi-backend   Start-VIBackend
Set-Alias vi-celery    Start-VICelery
Set-Alias vi-frontend  Start-VIFrontend
Set-Alias vi-redis     Start-VIRedis
Set-Alias vi-restart   Restart-VIBackend
Set-Alias vi-log       Show-VILog
Set-Alias pg-start     Start-VIPostgres
Set-Alias pg-stop      Stop-VIPostgres
Set-Alias pg-psql      Connect-VIPostgres

function Show-VICommands {
    Write-Host ''
    Write-Host 'FLUTTER-AI COMMANDS READY:' -ForegroundColor Cyan
    Write-Host '  Start-VIPostgres / Stop-VIPostgres / Connect-VIPostgres' -ForegroundColor Yellow
    Write-Host '  Start-VIRedis' -ForegroundColor Yellow
    Write-Host '  Start-VIBackend / Start-VICelery / Start-VIFrontend' -ForegroundColor Yellow
    Write-Host '  Start-VIAll   (everything, each in its own window)' -ForegroundColor Yellow
    Write-Host '  Restart-VIBackend / Restart-VICelery / Restart-VIFrontend, Show-VILog' -ForegroundColor Yellow
    Write-Host '  Short: vi-all vi-backend vi-celery vi-frontend vi-restart vi-log pg-start pg-stop' -ForegroundColor Yellow
    Write-Host ''
}
Show-VICommands
