## Air Writing Recognition

### Run (Windows, one command)

From the project root, run:

```powershell
.\run.ps1
```

If PowerShell blocks script execution, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

You can also double-click `run.bat`.

To run the camera benchmark directly, use the project interpreter:

```powershell
.\.venv\Scripts\python.exe .\camera_test.py
```

### What `run.ps1` does

1. Creates `.venv` if missing
2. Installs dependencies from `requirements.txt` if needed
3. Starts `main.py`

### FPS note

`main.py` is configured with an ultra-fast preset by default:

1. Camera capture at 320x240
2. Hand inference at 128x96
3. Inference throttled to ~8 updates/sec
4. Automatic camera backend selection (DirectShow/MSMF/Any), chooses fastest at startup

This improves responsiveness on weaker CPUs, with some accuracy tradeoff.
