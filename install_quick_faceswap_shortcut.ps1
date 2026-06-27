# Creates shortcuts for Quick Face Swap (Desktop + Favorites / Links bar).
# Right-click the shortcut -> Pin to taskbar to put it beside the Windows taskbar.

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $ProjectDir "QuickFaceSwap.bat"

if (-not (Test-Path $Launcher)) {
    Write-Error "Missing launcher: $Launcher"
}

$Wsh = New-Object -ComObject WScript.Shell

function New-Shortcut([string]$LinkPath) {
    $sc = $Wsh.CreateShortcut($LinkPath)
    $sc.TargetPath = $Launcher
    $sc.WorkingDirectory = $ProjectDir
    $sc.Description = "Pick source face and target image, then face-swap (PyEditorFromFFS conda env)"
    $sc.WindowStyle = 1
  # Face / photo icon from shell32
    $sc.IconLocation = "$env:SystemRoot\System32\imageres.dll,67"
    $sc.Save()
    Write-Host "Created: $LinkPath"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$Links = Join-Path $env:USERPROFILE "Links"

New-Shortcut (Join-Path $Desktop "Quick Face Swap.lnk")

if (-not (Test-Path $Links)) {
    New-Item -ItemType Directory -Path $Links -Force | Out-Null
}
New-Shortcut (Join-Path $Links "Quick Face Swap.lnk")

Write-Host ""
Write-Host "Done. To pin beside the taskbar:"
Write-Host "  1. Open Desktop or File Explorer -> Favorites (Links)"
Write-Host "  2. Right-click 'Quick Face Swap' -> Pin to taskbar"
Write-Host ""

