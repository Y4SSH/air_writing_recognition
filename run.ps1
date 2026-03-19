$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvDir = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements.txt"
$mainScript = Join-Path $projectRoot "main.py"
$installStamp = Join-Path $venvDir ".requirements_installed"

if (-not (Test-Path $venvPython)) {
    Write-Host "[setup] Creating virtual environment..."
    py -m venv $venvDir
}

if (-not (Test-Path $venvPython)) {
    throw "Failed to create virtual environment at $venvDir"
}

$needsInstall = -not (Test-Path $installStamp)
if (-not $needsInstall -and (Test-Path $requirements)) {
    $needsInstall = (Get-Item $requirements).LastWriteTimeUtc -gt (Get-Item $installStamp).LastWriteTimeUtc
}

if ($needsInstall) {
    Write-Host "[setup] Installing dependencies..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirements
    New-Item -Path $installStamp -ItemType File -Force | Out-Null
}

Write-Host "[run] Starting Air Writing Tracker..."
& $venvPython $mainScript
