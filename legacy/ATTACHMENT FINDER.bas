Attribute VB_Name = "ATTACHMENT FINDER"
Option Compare Database

Sub ExportRecordsWithAttachments()
    ' Declare necessary objects and variables
    Dim db As DAO.Database  ' Database object to access the current database
    Dim rs As DAO.Recordset ' Recordset to hold the result set of the query
    Dim rsAttach As DAO.Recordset ' Recordset to hold attachment data for each record
    Dim fso As Object  ' FileSystemObject to create and manage text files
    Dim txtFile As Object ' Object to write text to a file
    Dim outputPath As String ' Path where the output text file will be saved
    Dim folderPath As String ' Directory path to store the output file
    Dim hasAttachments As Boolean ' Flag to check if the record has any attachments
    
    ' Set your table and field names for easy configuration
    Const TableName As String = "ALTERED Device Inventory List_back_up_Nov13_18"  ' Change this to the appropriate table name
    Const IDField As String = "ID"  ' Change this to the field that contains the unique record ID
    Const AttachmentField As String = "Attached Report"  ' Change this to the field that contains attachments (if any)
    
    ' Set output file path where the result will be saved
    folderPath = "L:\Web-Based Traffic Calming\08 Human Resources\TT - Assigned Work\Archive\"  ' Directory where the text file will be stored
    outputPath = folderPath & "RecordsWithAttachments.txt"  ' Full path to the output text file

    ' Open the database and retrieve records from the specified table and fields
    Set db = CurrentDb  ' Get the current database
    Set rs = db.OpenRecordset("SELECT [" & IDField & "], [" & AttachmentField & "] FROM [" & TableName & "]", dbOpenDynaset)
    ' This opens a recordset with the ID and attachment fields from the specified table

    ' Create FileSystemObject to handle file creation
    Set fso = CreateObject("Scripting.FileSystemObject")

    ' Create a text file at the specified output path
    Set txtFile = fso.CreateTextFile(outputPath, True)  ' The "True" argument ensures that if the file exists, it will be overwritten
    
    ' Loop through the records in the recordset
    Do While Not rs.EOF  ' Continue looping until all records are processed
        hasAttachments = False  ' Reset the attachment flag for each record
        
        ' Check if the current record has an attachment (i.e., the attachment field is not null)
        If Not IsNull(rs(AttachmentField)) Then
            On Error Resume Next  ' Prevent errors if the record doesn't have an attachment field or the field is empty
            Set rsAttach = rs(AttachmentField).Value  ' Get the attachment sub-recordset (if any)
            On Error GoTo 0  ' Re-enable normal error handling after attempting to get the attachment
            
            ' If the attachment sub-recordset exists and contains records
            If Not rsAttach Is Nothing Then
                If rsAttach.RecordCount > 0 Then  ' Check if there are any records in the attachment sub-recordset
                    hasAttachments = True  ' Set the flag to True if attachments exist
                End If
                rsAttach.Close  ' Close the attachment sub-recordset
                Set rsAttach = Nothing  ' Clean up the attachment recordset object
            End If
        End If
        
        ' If the record has attachments, write the record ID to the output text file
        If hasAttachments Then
            txtFile.WriteLine rs(IDField)  ' Write the record ID to the file
        End If
        
        rs.MoveNext  ' Move to the next record in the recordset
    Loop  ' Repeat the loop until all records are processed
    
    ' Cleanup: close the recordset, file, and objects
    rs.Close  ' Close the main recordset
    Set rs = Nothing  ' Clean up the main recordset object
    txtFile.Close  ' Close the output text file
    Set txtFile = Nothing  ' Clean up the text file object
    Set fso = Nothing  ' Clean up the FileSystemObject
    Set db = Nothing  ' Clean up the database object
    
    ' Display a message box to confirm that the process is complete
    MsgBox "Processing complete. Output saved to: " & outputPath, vbInformation, "Done"
End Sub

