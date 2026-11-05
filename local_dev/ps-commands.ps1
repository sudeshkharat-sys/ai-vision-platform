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

function Show-VICommands {
    Write-Host ''
    Write-Host 'FLUTTER-AI COMMANDS READY:' -ForegroundColor Cyan
    Write-Host '  Start-VIPostgres / Stop-VIPostgres / Connect-VIPostgres' -ForegroundColor Yellow
    Write-Host '  Start-VIRedis' -ForegroundColor Yellow
    Write-Host '  Start-VIBackend / Start-VICelery / Start-VIFrontend' -ForegroundColor Yellow
    Write-Host '  Start-VIAll   (everything, each in its own window)' -ForegroundColor Yellow
    Write-Host ''
}
Show-VICommands
