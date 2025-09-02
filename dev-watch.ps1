# PowerShell Development Watch Script
# Automatically restarts services when code changes are detected

param(
    [string]$Service = "both",  # "backend", "frontend", or "both"
    [int]$DebounceSeconds = 3   # Wait time after detecting changes
)

Write-Host "🔍 Starting file watcher for Web Content Analyzer" -ForegroundColor Green
Write-Host "📁 Watching: $Service services" -ForegroundColor Yellow
Write-Host "⏱️  Debounce: $DebounceSeconds seconds" -ForegroundColor Yellow

# Define watch paths
$BackendPath = ".\backend"
$FrontendPath = ".\frontend"

# Create file system watchers
$Watchers = @()

if ($Service -eq "backend" -or $Service -eq "both") {
    $BackendWatcher = New-Object System.IO.FileSystemWatcher
    $BackendWatcher.Path = $BackendPath
    $BackendWatcher.Filter = "*.py"
    $BackendWatcher.IncludeSubdirectories = $true
    $BackendWatcher.EnableRaisingEvents = $true
    $Watchers += $BackendWatcher
    Write-Host "👀 Watching backend: $BackendPath" -ForegroundColor Cyan
}

if ($Service -eq "frontend" -or $Service -eq "both") {
    $FrontendWatcher = New-Object System.IO.FileSystemWatcher
    $FrontendWatcher.Path = $FrontendPath
    $FrontendWatcher.Filter = "*.py"
    $FrontendWatcher.IncludeSubdirectories = $true
    $FrontendWatcher.EnableRaisingEvents = $true
    $Watchers += $FrontendWatcher
    Write-Host "👀 Watching frontend: $FrontendPath" -ForegroundColor Cyan
}

# Track last restart time to implement debouncing
$LastRestart = Get-Date

# Define the restart action
$RestartAction = {
    param($Path, $ChangeType, $Name)
    
    $Now = Get-Date
    $TimeSinceLastRestart = ($Now - $script:LastRestart).TotalSeconds
    
    # Debounce: only restart if enough time has passed
    if ($TimeSinceLastRestart -gt $script:DebounceSeconds) {
        Write-Host "`n🔄 Change detected: $Name ($ChangeType)" -ForegroundColor Yellow
        Write-Host "📂 Path: $Path" -ForegroundColor Gray
        
        # Determine which service to restart
        if ($Path -like "*backend*") {
            Write-Host "🔧 Restarting backend..." -ForegroundColor Blue
            docker-compose restart backend
        } elseif ($Path -like "*frontend*") {
            Write-Host "🎨 Restarting frontend..." -ForegroundColor Magenta
            docker-compose restart frontend
        }
        
        $script:LastRestart = $Now
        Write-Host "✅ Service restarted!" -ForegroundColor Green
        Write-Host "⏳ Waiting for changes..." -ForegroundColor Gray
    }
}

# Register event handlers
foreach ($Watcher in $Watchers) {
    Register-ObjectEvent -InputObject $Watcher -EventName "Changed" -Action $RestartAction | Out-Null
    Register-ObjectEvent -InputObject $Watcher -EventName "Created" -Action $RestartAction | Out-Null
    Register-ObjectEvent -InputObject $Watcher -EventName "Renamed" -Action $RestartAction | Out-Null
}

Write-Host "`n✅ File watchers active!" -ForegroundColor Green
Write-Host "💡 Make changes to your code - services will auto-restart" -ForegroundColor Yellow
Write-Host "🛑 Press Ctrl+C to stop watching" -ForegroundColor Red
Write-Host "⏳ Waiting for changes...`n" -ForegroundColor Gray

# Keep the script running
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Cleanup watchers
    foreach ($Watcher in $Watchers) {
        $Watcher.EnableRaisingEvents = $false
        $Watcher.Dispose()
    }
    Write-Host "`n🛑 File watchers stopped" -ForegroundColor Red
}
