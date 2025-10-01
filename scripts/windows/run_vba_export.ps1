param(
  [Parameter(Mandatory=$true)][string]$Accdb,
  [Parameter(Mandatory=$true)][string]$MacroName,
  [Parameter(Mandatory=$true)][string]$OutJson
)
$cscript = "$env:SystemRoot\System32\cscript.exe"
$script = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "run_vba_export.vbs"
& $cscript //nologo $script $Accdb $MacroName $OutJson
exit $LASTEXITCODE
