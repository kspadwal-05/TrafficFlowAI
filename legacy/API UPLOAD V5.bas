Attribute VB_Name = "API UPLOAD V5"
Option Compare Database

Sub ProcessRecordsAndUpload()
    ' Declare database and recordset objects
    Dim db As DAO.Database
    Dim rs As DAO.Recordset
    
    ' FileSystemObject (FSO) is used for file/folder operations like checking if a folder exists
    Dim fso As Object
    
    ' Paths to where the subfolders and the output file are located
    Dim baseFolderPath As String
    Dim outputDir As String
    
    ' Strings to hold the complete JSON data
    Dim masterJSON As String
    Dim recordJSON As String
    
    ' The ID for each record from the database
    Dim recordID As String
    
    ' Set the folder where each record's files are stored (named after the record ID)
    baseFolderPath = "L:\Web-Based Traffic Calming\08 Human Resources\TT Winter 2025\TT- Assigned Work - Winter\Traffic Calming ETL\Bin IDs File Uploading\File Downloads 04-14-2025\"
    
    ' Set the folder where the final JSON output file will be saved
    outputDir = "L:\Web-Based Traffic Calming\08 Human Resources\TT Winter 2025\TT- Assigned Work - Winter\Traffic Calming ETL\Bin IDs File Uploading\Current BinIDs\"
    
    ' Create the FileSystemObject instance
    Set fso = CreateObject("Scripting.FileSystemObject")
    
    ' If the output folder doesn't exist, create it
    If Not fso.FolderExists(outputDir) Then
        fso.CreateFolder outputDir
    End If
    
    ' Begin building the master JSON object
    masterJSON = "{""RecordBins"": ["
    
    ' Open the database table and pull IDs for each record
    Set db = CurrentDb
    Set rs = db.OpenRecordset("SELECT [ID] FROM [ALTERED Device Inventory List_back_up_Nov13_18]", dbOpenDynaset)
    
    ' Loop through all the records in the recordset
    Do While Not rs.EOF
        recordID = rs!ID
        
        ' Construct the path to the folder for this record
        Dim recordFolderPath As String
        recordFolderPath = baseFolderPath & recordID & "\"
        
        ' Only process the record if its folder exists
        If fso.FolderExists(recordFolderPath) Then
            ' Upload the files in this folder and receive the JSON string of metadata
            recordJSON = UploadFilesInFolder(recordFolderPath)
            
            ' Add this record's JSON data to the master JSON string
            masterJSON = masterJSON & "{""record_id"":""" & recordID & """, ""uploads"":" & recordJSON & "},"
        Else
            ' Log folders that don't exist
            Debug.Print "Folder not found for RecordID: " & recordID
        End If
        
        ' Move to the next record in the table
        rs.MoveNext
    Loop
    
    ' Close and release recordset resources
    rs.Close
    Set rs = Nothing
    
    ' Remove trailing comma, if any, and close the JSON structure
    If Right(masterJSON, 1) = "," Then
        masterJSON = Left(masterJSON, Len(masterJSON) - 1)
    End If
    masterJSON = masterJSON & "]}"
    
    ' Save the master JSON to a file
    Dim outputFilePath As String
    outputFilePath = outputDir & "MasterBinIDs.json"
    
    ' Create the file and write the JSON content to it
    Dim jsonFile As Object
    Set jsonFile = fso.CreateTextFile(outputFilePath, True)
    jsonFile.Write masterJSON
    jsonFile.Close
    
    ' Show message when complete
    MsgBox "Processing complete. Master JSON file created at " & outputFilePath
End Sub

'---------------------------------------------------------
' Uploads up to 8 files from a folder to a web API using
' a multipart HTTP POST request and returns a JSON string
' describing the uploaded files and their associated BIN_IDs.
'---------------------------------------------------------
Function UploadFilesInFolder(folderPath As String) As String
    ' Objects for HTTP request, file handling, and upload data
    Dim http As Object
    Dim boundary As String
    Dim fso As Object, folder As Object, fileItem As Object
    Dim fileCount As Integer
    Dim url As String
    Dim colFiles As Collection
    Dim dictFile As Object
    
    ' API endpoint for uploading
    url = "https://was-intra-sit.toronto.ca/c3api_upload/upload/tci"
    
    ' This is a unique string used to separate parts in the multipart HTTP request
    boundary = "WebKitFormBoundarynNZRKgEd0ByxBIm5"
    
    ' Create the FSO and get the folder object
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set folder = fso.GetFolder(folderPath)
    
    ' Collection to hold metadata about each file
    Set colFiles = New Collection
    
    ' Create a binary stream to build the multipart HTTP request
    ' ADODB.Stream allows you to write binary data like files into memory
    Dim postStream As Object
    Set postStream = CreateObject("ADODB.Stream")
    postStream.Type = 1 ' Binary
    postStream.Open
    
    ' Variables for building HTTP headers
    Dim headerPart As String, headerBytes() As Byte
    Dim crlf() As Byte, fileBytes() As Byte
    Dim contentType As String
    crlf = StrConv(vbCrLf, vbFromUnicode) ' Line break in binary
    
    fileCount = 0
    ' Loop through each file in the folder (up to 8 files)
    For Each fileItem In folder.Files
        fileCount = fileCount + 1
        If fileCount > 8 Then Exit For
        
        ' Determine the content type based on file extension
        Select Case LCase(fso.GetExtensionName(fileItem.Name))
            Case "pdf": contentType = "application/pdf"
            Case "png": contentType = "image/png"
            Case "jpg", "jpeg": contentType = "image/jpeg"
            Case Else: contentType = "application/octet-stream"
        End Select
        
        ' Store file name and content type in a dictionary
        Set dictFile = CreateObject("Scripting.Dictionary")
        dictFile.Add "Name", fileItem.Name
        dictFile.Add "ContentType", contentType
        colFiles.Add dictFile
        
        ' Construct the multipart header for this file
        headerPart = "--" & boundary & vbCrLf
        headerPart = headerPart & "Content-Disposition: form-data; name=""file""; filename=""" & fileItem.Name & """" & vbCrLf
        headerPart = headerPart & "Content-Type: " & contentType & vbCrLf & vbCrLf
        headerBytes = StrConv(headerPart, vbFromUnicode)
        postStream.Write headerBytes
        
        ' Read the file contents and write to the postStream
        Dim fileStream As Object
        Set fileStream = CreateObject("ADODB.Stream")
        fileStream.Type = 1 ' Binary
        fileStream.Open
        fileStream.LoadFromFile fileItem.Path
        fileBytes = fileStream.Read
        fileStream.Close
        Set fileStream = Nothing
        
        postStream.Write fileBytes
        postStream.Write crlf ' Add a line break after file
    Next fileItem
    
    ' Add the closing boundary to signal end of multipart request
    Dim footerPart As String, footerBytes() As Byte
    footerPart = "--" & boundary & "--" & vbCrLf
    footerBytes = StrConv(footerPart, vbFromUnicode)
    postStream.Write footerBytes
    
    ' Rewind stream and read all the binary content into a variable
    postStream.Position = 0
    Dim postData() As Byte
    postData = postStream.Read
    postStream.Close
    Set postStream = Nothing
    
    ' Send the request via HTTP
    Set http = CreateObject("MSXML2.XMLHTTP")
    http.Open "POST", url, False
    http.setRequestHeader "Content-Type", "multipart/form-data; boundary=" & boundary
    http.setRequestHeader "Host", "was-intra-sit.toronto.ca"
    http.setRequestHeader "Origin", "https://was-intra-sit.toronto.ca"
    
    Dim vPostData As Variant
    vPostData = postData
    http.send vPostData

    ' Default return in case of error
    Dim finalJSON As String
    finalJSON = "[]"
    
    ' Handle a successful HTTP response
    If http.Status = 200 Then
        Dim responseJSON As String
        responseJSON = http.responseText
        
        Dim json As Object
        Set json = JsonConverter.ParseJson(responseJSON)
        
        ' Check if BIN_ID array exists in response
        If Not json.exists("BIN_ID") Then
            UploadFilesInFolder = "[]"
            Exit Function
        End If
        
        Dim binIDArray As Object
        Set binIDArray = json("BIN_ID")
        
        ' Start building JSON array of file info
        finalJSON = "["
        Dim i As Integer
        Dim dictFileInfo As Object
        
        ' Match each local file with its API response
        For i = 1 To colFiles.Count
            Set dictFileInfo = colFiles(i)
            finalJSON = finalJSON & "{""filename"": """ & dictFileInfo("Name") & """, "
            finalJSON = finalJSON & """contentType"": """ & dictFileInfo("ContentType") & """, "
            finalJSON = finalJSON & """BIN_ID"": ["
            
            Dim item As Object
            For Each item In binIDArray
                If Right(item("file_name"), Len(dictFileInfo("Name"))) = dictFileInfo("Name") Then
                    finalJSON = finalJSON & "{""bin_id"": """ & item("bin_id") & """, ""file_name"": """ & item("file_name") & """},"
                End If
            Next item
            
            ' Trim trailing comma and close object
            If Right(finalJSON, 1) = "," Then
                finalJSON = Left(finalJSON, Len(finalJSON) - 1)
            End If
            finalJSON = finalJSON & "]},"
        Next i
        
        ' Final cleanup of JSON string
        If Right(finalJSON, 1) = "," Then
            finalJSON = Left(finalJSON, Len(finalJSON) - 1)
        End If
        finalJSON = finalJSON & "]"
    Else
        Debug.Print "Upload failed for folder: " & folderPath & " - " & http.Status & " " & http.statusText
    End If
    
    ' Cleanup and return JSON result
    Set http = Nothing
    UploadFilesInFolder = finalJSON
End Function
