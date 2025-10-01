Attribute VB_Name = "FILE DOWNLOADING"
Option Compare Database

Sub FileDownload()
    ' Declare necessary objects and variables
    Dim db As DAO.Database  ' Object for accessing the current database
    Dim rs As DAO.Recordset ' Main recordset to hold data from the "Device Inventory List" table
    Dim rsAttach As DAO.Recordset ' Recordset for attachments associated with each record
    Dim folderName As String  ' String to hold the path for the folder to store attachments
    Dim filename As String  ' String to store the filename of the attachment
    Dim filePath As String  ' String to hold the complete path where the attachment will be saved
    
    ' Error handling: if an error occurs, the code will jump to ErrorHandler
    On Error GoTo ErrorHandler

    ' Set the database and recordset objects
    Set db = CurrentDb()  ' Access the current database
    Set rs = db.OpenRecordset("Device Inventory List_back_up_Nov13_18")  ' Open the specified table as a recordset

    ' Loop through each record in the main recordset
    Do While Not rs.EOF  ' Continue looping until the end of the recordset
        ' Create a folder based on the value of the "ID" field (e.g., folder named after record ID)
        folderName = CurrentProject.Path & "\" & rs.Fields("ID").Value
        ' Check if the folder already exists; if not, create it
        If Dir(folderName, vbDirectory) = "" Then
            MkDir folderName  ' Create the folder if it doesn't already exist
        End If
        
        ' Check if there are any attachments in the "Attached Report" field for the current record
        If Not IsNull(rs.Fields("Attached Report").Value) Then
            ' If attachments exist, open the sub-recordset containing the attachment data
            Set rsAttach = rs.Fields("Attached Report").Value  ' Get the attachment recordset
            ' Loop through each attachment in the sub-recordset
            Do While Not rsAttach.EOF
                ' Get the filename of the attachment
                filename = rsAttach.Fields("FileName").Value
                ' Construct the full file path where the attachment will be saved
                filePath = folderName & "\" & filename
                ' Save the attachment to the constructed file path
                rsAttach.Fields("FileData").SaveToFile filePath
                ' Move to the next attachment in the sub-recordset
                rsAttach.MoveNext
            Loop
            ' Close the attachment recordset after processing all attachments
            rsAttach.Close
        End If
        
        ' Move to the next record in the main recordset
        rs.MoveNext
    Loop

    ' Once all records are processed, show a success message
    MsgBox "Attachments Downloaded Successfully!", vbInformation

ExitProcedure:
    ' Cleanup: close the recordsets and release the objects
    If Not rs Is Nothing Then rs.Close  ' Close the main recordset
    If Not rsAttach Is Nothing Then rsAttach.Close  ' Close the attachment recordset (if open)
    Set rs = Nothing  ' Release the main recordset object
    Set rsAttach = Nothing  ' Release the attachment recordset object
    Set db = Nothing  ' Release the database object
    Exit Sub  ' Exit the procedure

ErrorHandler:
    ' If an error occurs, display the error message and jump to cleanup section
    MsgBox "Error: " & Err.Description, vbCritical
    Resume ExitProcedure  ' Resume execution at ExitProcedure to ensure cleanup occurs
End Sub

