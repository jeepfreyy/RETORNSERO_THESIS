# ─────────────────────────────────────────────────────────────────────────────
#  BARANGAY SENTINEL — One-click server startup
#  Starts Flask + Cloudflare tunnel, auto-updates the mobile app URL
# ─────────────────────────────────────────────────────────────────────────────

$CLOUDFLARED  = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$PROJECT_ROOT = "E:\RETORNSERO_THESIS-main"
$CONSTANTS    = "$PROJECT_ROOT\mobile\lib\constants.ts"
$TUNNEL_LOG   = "$env:TEMP\cf_tunnel.log"
$FLASK_LOG    = "$env:TEMP\flask_server.log"
$FLASK_ERR    = "$env:TEMP\flask_server_err.log"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BARANGAY SENTINEL — Starting Server  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Stop any previous instances ───────────────────────────────────────────────
Write-Host "[1/4] Stopping old instances..." -ForegroundColor Yellow
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# ── Start Flask ───────────────────────────────────────────────────────────────
Write-Host "[2/4] Starting Flask server..." -ForegroundColor Yellow
Start-Process -FilePath "python" `
  -ArgumentList "app.py" `
  -WorkingDirectory $PROJECT_ROOT `
  -RedirectStandardOutput $FLASK_LOG `
  -RedirectStandardError  $FLASK_ERR `
  -WindowStyle Hidden

# Wait for Flask to be ready
$flaskReady = $false
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Seconds 1
  $conn = Test-NetConnection -ComputerName localhost -Port 5001 -InformationLevel Quiet -WarningAction SilentlyContinue
  if ($conn) { $flaskReady = $true; break }
  Write-Host "   Waiting for Flask... ($i/20)" -ForegroundColor DarkGray
}

if (-not $flaskReady) {
  Write-Host "❌ Flask failed to start. Check your app.py for errors." -ForegroundColor Red
  Get-Content $FLASK_ERR -ErrorAction SilentlyContinue | Select-Object -Last 10
  Read-Host "Press Enter to exit"
  exit 1
}
Write-Host "   ✅ Flask is running on port 5001" -ForegroundColor Green

# ── Start Cloudflare tunnel ───────────────────────────────────────────────────
Write-Host "[3/4] Starting Cloudflare tunnel..." -ForegroundColor Yellow
Remove-Item $TUNNEL_LOG -ErrorAction SilentlyContinue
Start-Process -FilePath $CLOUDFLARED `
  -ArgumentList "tunnel --url http://localhost:5001 --logfile `"$TUNNEL_LOG`"" `
  -WindowStyle Hidden

# Wait for tunnel URL to appear in log
$tunnelUrl = $null
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 1
  if (Test-Path $TUNNEL_LOG) {
    $match = Select-String -Path $TUNNEL_LOG -Pattern "https://[a-z0-9\-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue
    if ($match) {
      $tunnelUrl = ($match.Matches[0].Value)
      break
    }
  }
  Write-Host "   Waiting for tunnel URL... ($i/30)" -ForegroundColor DarkGray
}

if (-not $tunnelUrl) {
  Write-Host "❌ Cloudflare tunnel failed to start." -ForegroundColor Red
  Read-Host "Press Enter to exit"
  exit 1
}
Write-Host "   ✅ Tunnel is live: $tunnelUrl" -ForegroundColor Green

# ── Update mobile app constants.ts ────────────────────────────────────────────
Write-Host "[4/4] Updating mobile app URL..." -ForegroundColor Yellow

$hostname = $tunnelUrl -replace "https://", ""
$content  = Get-Content $CONSTANTS -Raw
$updated  = $content `
  -replace "export const SERVER_IP\s*=\s*'[^']*'",       "export const SERVER_IP   = '$hostname'" `
  -replace "export const SERVER_PORT\s*=\s*'[^']*'",     "export const SERVER_PORT = '443'" `
  -replace "export const BASE_URL\s*=\s*'[^']*'",        "export const BASE_URL    = '$tunnelUrl'"
Set-Content -Path $CONSTANTS -Value $updated -Encoding UTF8
Write-Host "   ✅ constants.ts updated" -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅  SERVER IS LIVE                   " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Web Dashboard  : $tunnelUrl" -ForegroundColor Cyan
Write-Host "  Mobile API     : $tunnelUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "  1. Run: cd mobile && npx expo start" -ForegroundColor White
Write-Host "  2. Scan QR code with Expo Go" -ForegroundColor White
Write-Host "  3. Share the URL above for web access" -ForegroundColor White
Write-Host ""
Write-Host "  Press Ctrl+C or close this window to stop the server." -ForegroundColor DarkGray
Write-Host ""

# ── Keep running until user closes ───────────────────────────────────────────
try {
  while ($true) { Start-Sleep -Seconds 60 }
} finally {
  Write-Host ""
  Write-Host "Stopping server..." -ForegroundColor Yellow
  Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
  Get-Process python       -ErrorAction SilentlyContinue | Stop-Process -Force
  Write-Host "Server stopped. Goodbye!" -ForegroundColor Green
}
