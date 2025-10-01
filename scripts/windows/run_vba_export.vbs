Option Explicit
' run_vba_export.vbs
' Usage: cscript //nologo run_vba_export.vbs "C:\path\db.accdb" "ExportToJSON" "C:\out\export_from_vba.json"
Dim accdb, macroName, outPath
If WScript.Arguments.Count < 3 Then
  WScript.Echo "Usage: cscript //nologo run_vba_export.vbs <accdb> <macroName> <outJson>"
  WScript.Quit 1
End If
accdb = WScript.Arguments(0)
macroName = WScript.Arguments(1)
outPath = WScript.Arguments(2)
Dim shell, cmd
Set shell = CreateObject("WScript.Shell")
cmd = """" & "msaccess.exe" & """" & " " & """" & accdb & """" & " /x " & """" & macroName & """" & " /cmd " & """" & outPath & """"
WScript.Echo "Running: " & cmd
Dim rc: rc = shell.Run(cmd, 1, True)
WScript.Echo "Access returned: " & rc
WScript.Quit rc
