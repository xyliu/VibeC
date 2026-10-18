Set ws = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)

pythonwExe = currentDir & "\.venv\Scripts\pythonw.exe"
scriptPath = currentDir & "\demo_qwen3_asr.py"

If Not fso.FileExists(pythonwExe) Then
    MsgBox "Python environment not found.", 16, "VibeC Error"
    WScript.Quit 1
End If

ws.CurrentDirectory = currentDir
ws.Run """" & pythonwExe & """ """ & scriptPath & """ --amd-npu", 0, False
