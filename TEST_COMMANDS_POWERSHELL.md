# End-to-end test commands (Windows PowerShell)

Live-camera runs against the YOLO26 detector. Run from the repo root, and
paste each command as ONE line -- a terminal that wraps it onto several lines
breaks it.

Every `perceive.py` command needs `--config perception\config\webcam.json`.
Without it the script falls back to `zones.example.json`, which runs YOLOX,
not YOLO26.

## Setup

```powershell
if (-not (Test-Path identity\data\employees.csv)) { Copy-Item identity\data\employees.example.csv identity\data\employees.csv }
python identity\enroll.py
```

`enroll.py` is interactive: person ID (e.g. `EMP-0001`), optional display
name, then photo paths one per line, blank line to finish.

## Cases

### 1. Object detection
Hold a phone at chest height, back toward the camera.
```powershell
cmd /c "python perception\perceive.py --config perception\config\webcam.json --objects --max-frames 150 2> logs\case1-objects.stderr.log > logs\case1-objects.jsonl"
```

### 2. Track ID on the detected person
Sit still in the right half of the picture.
```powershell
cmd /c "python perception\perceive.py --config perception\config\webcam.json --max-frames 150 2> logs\case2-track.stderr.log > logs\case2-track.jsonl"
```

### 3. Authentication against a known user
Face the camera, both eyes visible.
```powershell
cmd /c "python perception\perceive.py --config perception\config\webcam.json --identity identity\config\identity.example.json --max-frames 200 2> logs\case3-auth-known.stderr.log > logs\case3-auth-known.jsonl"
```

### 3a. Unknown users
An unenrolled person in frame, facing the camera; you out of frame.
```powershell
cmd /c "python perception\perceive.py --config perception\config\webcam.json --identity identity\config\identity.example.json --max-frames 200 2> logs\case3a-auth-unknown.stderr.log > logs\case3a-auth-unknown.jsonl"
```

### 4 and 5. Person ID returned, name resolved from the CSV
Face the camera. The name appears only on the preview label (`trk-0001 0.9x Sashank`),
never in the log -- events carry the ID by design.
```powershell
cmd /c "python perception\perceive.py --config perception\config\webcam.json --identity identity\config\identity.example.json --preview --max-frames 200 2> logs\case4-5-personid-name.stderr.log > logs\case4-5-personid-name.jsonl"
```

### 6. Zone detection
In the orange zone ~10 s, out of it for a full ~10 s, back in and sit still.
Do not press `q`.
```powershell
cmd /c "python perception\perceive.py --config perception\config\webcam.json --identity identity\config\identity.example.json --preview --max-frames 300 2> logs\case6-verify.stderr.log > logs\case6-verify.jsonl"
```

### 7, 7a, 7b. Action detection
Whole body in frame, head to ankles -- tilt or lower the laptop and stand
~3-4 m back. Identify up close facing the lens first, walk straight back
without leaving frame, wave one hand above your head for ~25 s, then stand
still. Do not press `q`.
```powershell
cmd /c "python perception\perceive.py --config perception\config\webcam.json --identity identity\config\identity.example.json --actions --preview --max-frames 500 2> logs\case7-action.stderr.log > logs\case7-action.jsonl"
```

## Supporting commands

Schema check on any log:
```powershell
python perception\tools\validate_events.py logs\case6-verify.jsonl
```

Diagnostic for the false person track (logs every box, saves a contact sheet;
`preview_pose.py` has its own `--log`, so no redirection):
```powershell
python perception\tools\preview_pose.py --detector --config perception\config\webcam.json --max-frames 150 --no-window --save logs\phantom-diag.png --sheet logs\phantom-diag-sheet.png --sheet-every 2 --log logs\phantom-diag.jsonl
```

Unit tests -- regressions only, not evidence for the cases above:
```powershell
foreach ($t in "perception\test_perception.py","perception\test_stages.py","perception\test_intervals.py","perception\test_objects.py","perception\test_ppe.py","perception\test_yolo26.py","identity\test_identity.py","identity\test_directory.py","identity\test_alignment.py","harness\test_engine.py","harness\test_lint.py") { python $t *> $null; if ($LASTEXITCODE -eq 0) { "pass $t" } else { "FAIL $t" } }
```

## Why the commands look like this

- Windows PowerShell 5.1 writes UTF-16 for `>`, `2>` and `Tee-Object`, and
  re-encodes text piped between two programs. JSON readers reject the result.
- The `>` and `2>` inside `cmd /c "..."` are cmd's, which write Python's bytes
  unchanged. Keep them inside the quotes; do not replace `cmd /c` with
  `| Tee-Object`.
- Nothing prints to the screen during a run. To watch events arrive, use a
  second PowerShell window (read-only, safe):
  `Get-Content logs\case6-verify.jsonl -Wait`
