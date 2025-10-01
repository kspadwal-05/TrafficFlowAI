Attribute VB_Name = "JSON GENERATOR V7"
Option Compare Database ' Ensures text comparisons are case-insensitive

Sub JSONTESTERV7()
    ' Declare database and recordset objects to interact with Access tables
    Dim db As DAO.Database
    Dim rs As DAO.Recordset

    ' Will hold the final JSON output string
    Dim json As String
    
    ' File path to save the generated JSON
    Dim jsonFile As String
    
    ' Integer for assigning a free file handle when working with file I/O
    Dim fileNum As Integer
    
    ' The name of the table to be exported to JSON (can be changed for testing)
    Dim TableName As String
    TableName = "ALTERED Device Inventory List_back_up_Nov13_18"

    ' Loop/control variables
    Dim i As Integer
    Dim exportedCount As Long   ' Track how many records are successfully exported
    Dim skippedCount As Long    ' Track how many records are skipped
    exportedCount = 0
    skippedCount = 0

    ' Tracks how many records have an empty street name
    Dim emptyStreetCount As Integer
    emptyStreetCount = 0

    ' --- Define field mappings between the Access table fields and JSON keys ---

    ' These are the field names from the Access table that represent the main property data
    Dim propertyFields As Variant
    propertyFields = Array("ID", "Location - Street Name1", "From - Street Name2", "To - Street Name3", _
                           "Road Classification", "Date Data Requested", "Date Data Received", "Street Operation", _
                           "Volume (vpd)", "Posted Speed Limit (km/h)", "Average Speed (km/h)", _
                           "85th Percentile Speed (km/h)", "Staff Recommended2", "Plan/Drawing Number", _
                           "Estimated Cost", "Comments")

    ' These are the keys the fields will map to in the JSON output
    Dim jsonFields As Variant
    jsonFields = Array("recordId", "streetName", "intersection1", "intersection2", "roadType", _
                       "requestedAnalysisInfoDate", "receivedAnalysisInfoDate", "streetOperation", "volume", _
                       "postedSpeedLimit", "averageSpeed", "percentileSpeed85", "analysisRecommended", _
                       "planNumber", "estimatedCost", "comments")

    ' --- Speed control device fields from the table (group 1) ---
    Dim deviceFields1 As Variant
    deviceFields1 = Array("Speed Humps", "Laneway Speed Bump")

    ' Corresponding JSON keys for deviceFields1
    Dim jsonDeviceFields1 As Variant
    jsonDeviceFields1 = Array("numSpeedHumps", "numSpeedBumps")

    ' --- Other legacy traffic calming device fields from the table (group 2) ---
    Dim deviceFields2 As Variant
    deviceFields2 = Array("Chicanes", "Gateways", "Intersection Narrowings", "Mid-Block Pinch Points", _
                          "Raised Center Medians", "Raised Crosswalks", "Raised Intersections", "Traffic Circles")

    ' Corresponding JSON keys for deviceFields2
    Dim jsonDeviceFields2 As Variant
    jsonDeviceFields2 = Array("legacy_numChicanes", "legacy_numGateways", "legacy_numIntersectionNarrowings", _
                              "legacy_numMidBlockPinchPoints", "legacy_numRaisedCenterMedians", "legacy_numCrosswalks", _
                              "legacy_numRaisedIntersections", "legacy_numTrafficCircles")

    ' --- Council report-related fields (2 sets for 2 reports) ---
    Dim report1Fields As Variant
    report1Fields = Array("Clause Number1", "Council Approval of 1st Report", "1st Recommendation Amended 2", _
                          "First Report Date To CC", "1st Report Decision Hyperlink")

    Dim report2Fields As Variant
    report2Fields = Array("Clause Number2", "Final Council Approval", "Recommendations Amended 2", _
                          "Second Report Date to Standing Committee", "2nd Report Decision HyperLink")

    ' Common JSON key mapping for both reports
    Dim jsonReportFields As Variant
    jsonReportFields = Array("ccAgendaItem", "ccApproval", "ccRecommendation", "ccReportDate", "ccReportLink")

    ' --- Fields related to the removal of traffic calming devices ---
    Dim removeFields As Variant
    removeFields = Array("Remvoal Contract Num", "Date of Removal", "Remvoal Reason", "Removed")

    ' Corresponding JSON keys for removal fields
    Dim jsonRemoveFields As Variant
    jsonRemoveFields = Array("removalContractNumber", "removalDate", "removalReason", "removed")

    ' Open the database and the specified table
    Set db = CurrentDb
    Set rs = db.OpenRecordset(TableName) ' Load records from the table into the recordset

    ' Define the output path and file name for the JSON file
    jsonFile = CurrentProject.Path & "\" & TableName & ".json"

    ' Create an ADODB Stream object for writing text to the file with UTF-8 encoding
    Dim fileStream As Object
    Set fileStream = CreateObject("ADODB.Stream")
    fileStream.Charset = "utf-8"
    fileStream.Open

    ' Begin the JSON array
    fileStream.WriteText "[", 1 ' The "1" is for writing a new line (optional formatting)

    ' Position the recordset to the first record
    rs.MoveFirst

    ' Boolean flag to help insert commas correctly between JSON objects
    Dim isFirstRecord As Boolean
    isFirstRecord = True

    Do While Not rs.EOF ' While the recordset has not reached end of file
        ' Check for empty street fields before proceeding
        Dim locationValue As String
        locationValue = Nz(rs.Fields("Location - Street Name1").Value, "")
        Dim fromValue As String
        fromValue = Nz(rs.Fields("From - Street Name2").Value, "")
        Dim toValue As String
        toValue = Nz(rs.Fields("To - Street Name3").Value, "")
        
        ' Check if any of the required street fields are empty
        If Len(locationValue) = 0 Or Len(fromValue) = 0 Or Len(toValue) = 0 Then
            skippedCount = skippedCount + 1 ' Count the skipped records with missing street fields
        Else
            If Not isFirstRecord Then
                fileStream.WriteText "", 1
            Else
                isFirstRecord = False
            End If
        
            json = "{"
            json = json & """status"": """ & rs.Fields("Status").Value & """, "
            
            For i = LBound(propertyFields) To UBound(propertyFields)   'Loop through property fields
                Dim fieldName As String
                fieldName = propertyFields(i)
               
                If Not IsNull(rs.Fields(fieldName)) Then
                    Dim temp As String
                    ' Check if the field contains a date. If so, format it.
                    If IsDate(rs.Fields(fieldName).Value) Then
                        temp = Format(rs.Fields(fieldName).Value, "yyyy-mm-dd")
                    Else
                        temp = rs.Fields(fieldName).Value
                        If i = 1 Or i = 2 Or i = 3 Then
                            temp = GetStandardizedAddress(temp)
                        End If
                    End If
                    temp = Replace(temp, vbCrLf, "") 'Remove line feeds and carriage returns
                    temp = Replace(temp, vbCr, "")
                    temp = Replace(temp, vbLf, "")
                    temp = Replace(temp, Chr(9), "")
                    temp = Replace(temp, Chr(10), "")
                    temp = Replace(temp, Chr(13), "")
                    temp = Replace(temp, Chr(0), "")
                    temp = Replace(temp, ChrW(&HFFFD), "")
                    temp = Replace(temp, """", "\""")
                    json = json & """" & jsonFields(i) & """: """ & temp & """, "
                Else
                    json = json & """" & jsonFields(i) & """: """", "
                End If
            Next i
            
            For i = LBound(deviceFields1) To UBound(deviceFields1)
                fieldName = deviceFields1(i)
               
                If Not IsNull(rs.Fields(fieldName)) Then
                    json = json & """" & jsonDeviceFields1(i) & """: """ & rs.Fields(fieldName).Value & """"
                Else
                    json = json & """" & jsonDeviceFields1(i) & """: ""0"""
                End If
               
                json = json & ", "
            Next i
            
            ' Static values for fields not present in the db
            json = json & """requestTrackingNumber"": """", "
            json = json & """percentileSpeed95"": """", "
            json = json & """locationWarranted"": """", "
            json = json & """reportType"": """", "
            json = json & """initialSpeedCushions"": """", "
            json = json & """initialSpeedHumps"": """", "
            json = json & """initialSpeedBumps"": """", "
            json = json & """otherDeviceInformation"": """", "
            json = json & """numSpeedCushions"": """", "
            json = json & """numIslands"": """", "
            
            Dim otherDeviceTotal As Long
            Dim otherDeviceTypes As String
            Dim fieldValue As Long
            otherDeviceTotal = 0
            otherDeviceTypes = "["

            For i = LBound(deviceFields2) To UBound(deviceFields2)
                fieldName = deviceFields2(i)
                
                ' Set fieldValue correctly, assuming it's retrieved from somewhere
                ' (e.g., from a corresponding value array or another logic source)
                If Not IsNull(rs.Fields(fieldName)) Then
                    fieldValue = rs.Fields(fieldName).Value
                Else
                    fieldValue = 0
                End If
                
                ' Check if fieldValue is greater than 0, and only then process the item
                If fieldValue > 0 Then
                    ' Add to the total sum
                    otherDeviceTotal = otherDeviceTotal + fieldValue
                    
                    ' Add the device type to the array (with correct formatting)
                    If Len(otherDeviceTypes) > 1 Then
                        ' Add a comma if this is not the first element
                        otherDeviceTypes = otherDeviceTypes & ", "
                    End If
                    
                    ' Append the device to the JSON array string
                    otherDeviceTypes = otherDeviceTypes & "{""type"": """ & Replace(jsonDeviceFields2(i), "legacy_", "") & """, ""value"": """ & fieldValue & """}"
                End If
            Next i
            
            otherDeviceTypes = otherDeviceTypes & "]"
            
            json = json & """otherDeviceTotal"": """ & otherDeviceTotal & """, "
            json = json & """otherDeviceTypes"": " & otherDeviceTypes & ", "
            json = json & """priorityRanking"": """ & rs.Fields("Ranking").Value & """, "
            
            json = json & """legacy_data"":" & "{"
            
            For i = LBound(deviceFields2) To UBound(deviceFields2)
                fieldName = deviceFields2(i)
               
                If Not IsNull(rs.Fields(fieldName)) Then
                    json = json & """" & jsonDeviceFields2(i) & """: """ & rs.Fields(fieldName).Value & """"
                Else
                    json = json & """" & jsonDeviceFields2(i) & """: ""0"""
                End If
               
                json = json & ", "
            Next i
            
            'old database fields
            Dim oldFields As Variant
            oldFields = Array("newid", "Initiator", "Request Received", "Semi-formal Initiation", "Investigator", "Ward Number/Name", "Councillors", "Community Council", "Community Council 2018", "Petition Waived by Council", "Community Safety Zone", "Petition Waived by GM", "Traffic Plan Requested", "Traffic Plan Received", "Staff Recommended", "Staff Recommended(YES/NO)", "Actual Cost", "Account Number", "Budget Year", "1st Recommendations Amended", "1st Recommendations Amended(YES/NO)", "New Polling Waived", "Date Poll Sent Out", "Date Poll Closed", "% Poll Response", "% Support", "Negative Poll", "Poll Result 2", "Poll Result", "EA 1st Notice of Commencement", "EA 2nd Notice of Commencement", "Start Road Alteration Advertising", "End Road Alteration Advertising", "Second Report Date to CC", "Council Approved", "Recommendations Amended", "Recommendations Amended(YES/NO)", "EA Notice of Completion Open", "EA Notice of Completion Closed", "Field Marks - Received", "Field Marks - Marked By")
            Dim oldFields2 As Variant
            oldFields2 = Array("Field Marks - Completed", "Signage Date - Requested", "Construction Details Referred To", "Construction Details Received", "Construction Details Consideration", "Construction Details Section", "Person Notified", "Date of Notification", "Frmr CoT Point Value", "Number of INJ Collisions", "Number of PDO Collisions", "Number of Ped & Bike Factors", "Old Ward Number", "Old Ward Name", "New Ward Name", "New Ward #-Name 2018", "Old Community Council", "Warrant 1", "Warrant 2", "Warrant 3", "Total Number of Devices", "Position", "First Report Date to Standing Committee", "2nd Recommendations Amended", "2nd Recommendations Amended(YES/NO)", "Maintenance Year", "Ward2018", "NewDist")
            
            For i = LBound(oldFields) To UBound(oldFields)
                fieldName = oldFields(i)
                If Not IsNull(rs.Fields(fieldName)) Then
                    If IsDate(rs.Fields(fieldName).Value) Then
                        json = json & """oldDB_" & Replace(fieldName, " ", "") & """: """ & Format(rs.Fields(fieldName).Value, "yyyy-mm-dd") & """, "
                    Else
                        json = json & """oldDB_" & Replace(fieldName, " ", "") & """: """ & cleanString(rs.Fields(fieldName).Value) & """, "
                    End If
                Else
                    json = json & """oldDB_" & Replace(fieldName, " ", "") & """: """", "
                End If
            Next i
            
            For i = LBound(oldFields2) To UBound(oldFields2)
                fieldName = oldFields2(i)
                If Not IsNull(rs.Fields(fieldName)) Then
                    If IsDate(rs.Fields(fieldName).Value) Then
                        json = json & """oldDB_" & Replace(fieldName, " ", "") & """: """ & Format(rs.Fields(fieldName).Value, "yyyy-mm-dd") & """"
                    Else
                        json = json & """oldDB_" & Replace(fieldName, " ", "") & """: """ & cleanString(rs.Fields(fieldName).Value) & """"
                    End If
                Else
                    json = json & """oldDB_" & Replace(fieldName, " ", "") & """: """""
                End If
                If i < UBound(oldFields2) Then
                    json = json & ", "
                End If
            Next i
            json = json & "}, "
            
            ' Check if all ccReportItems are empty
            Dim isEmptyCCReportItems As Boolean
            isEmptyCCReportItems = True ' Assume it's empty unless proven otherwise
            
            For i = LBound(report1Fields) To UBound(report1Fields)
                If Not IsNull(rs.Fields(report1Fields(i))) And Len(rs.Fields(report1Fields(i)).Value) > 0 Then
                    isEmptyCCReportItems = False ' Found a non-empty field
                    Exit For
                End If
            Next i
            
            For i = LBound(report2Fields) To UBound(report2Fields)
                If Not IsNull(rs.Fields(report2Fields(i))) And Len(rs.Fields(report2Fields(i)).Value) > 0 Then
                    isEmptyCCReportItems = False ' Found a non-empty field
                    Exit For
                End If
            Next i
            
            ' If ccReportItems is empty, set it to an empty array, otherwise proceed with the normal logic
            If isEmptyCCReportItems Then
                json = json & """ccReportItems"": [], "
            Else
                json = json & """ccReportItems"":" & "["
                Dim report As Variant
                For num = 1 To 2
                    ' Select the proper report fields
                    report = IIf(num <> 1, report1Fields, report2Fields)
                    json = json & "{"
                    ' Format the By-law Approval Date if it is a date
                    If Not IsNull(rs.Fields("By-law Approval Date").Value) And IsDate(rs.Fields("By-law Approval Date").Value) Then
                        json = json & """bylawApprovalDate"": """ & Format(rs.Fields("By-law Approval Date").Value, "yyyy-mm-dd") & """, "
                    Else
                        json = json & """bylawApprovalDate"": """", "
                    End If
                    
                    Dim temp1 As Variant
                    temp1 = rs.Fields("By-law Number")
                    
                    If Not IsNull(temp1) Then
                        temp1 = Replace(temp1, vbCrLf, "")
                        temp1 = Replace(temp1, vbCr, "")
                        temp1 = Replace(temp1, vbLf, "")
                        temp1 = Replace(temp1, Chr(9), "")
                        temp1 = Replace(temp1, Chr(10), "")
                        temp1 = Replace(temp1, Chr(13), "")
                        temp1 = Replace(temp1, Chr(0), "")
                        temp1 = Replace(temp1, ChrW(&HFFFD), "")
                        temp1 = Replace(temp1, """", "\""")
                    End If
                    
                    json = json & """bylawNumber"": """ & temp1 & """, "
                    json = json & """ccReportType"": """", "
                    json = json & """councilDeclined"": """", "
                    
                    For i = LBound(jsonReportFields) To UBound(jsonReportFields)
                        fieldName = report(i)
                        
                        If Not IsNull(rs.Fields(fieldName)) Then
                            Dim temp2 As Variant
                            If IsDate(rs.Fields(fieldName).Value) Then
                                temp2 = Format(rs.Fields(fieldName).Value, "yyyy-mm-dd")
                            Else
                                temp2 = RemoveHash(rs.Fields(fieldName).Value)
                            End If
                            temp2 = Replace(temp2, """", "\""")
                            temp2 = Replace(temp2, vbCrLf, "")
                            temp2 = Replace(temp2, vbCr, "")
                            temp2 = Replace(temp2, vbLf, "")
                            temp2 = Replace(temp2, Chr(9), "")
                            temp2 = Replace(temp2, Chr(10), "")
                            temp2 = Replace(temp2, Chr(13), "")
                            temp2 = Replace(temp2, Chr(0), "")
                            temp2 = Replace(temp2, ChrW(&HFFFD), "")
                            
                            json = json & """" & jsonReportFields(i) & """: """ & temp2 & """"
                        Else
                            ' Removed extra comma here; we'll add a comma later if needed.
                            json = json & """" & jsonReportFields(i) & """: """""
                        End If
                        If i < UBound(jsonReportFields) Then
                            json = json & ", "
                        End If
                    Next i
                    json = json & "}"
                    If num <> 2 Then
                        json = json & ", "
                    End If
                Next num
                json = json & "], "
            End If
            
            ' For initial installation/completion dates, check if they are dates and format accordingly
            If Not IsNull(rs.Fields("Signage Date - Installed").Value) And IsDate(rs.Fields("Signage Date - Installed").Value) Then
                json = json & """initialInstallationDate"": """ & Format(rs.Fields("Signage Date - Installed").Value, "yyyy-mm-dd") & """, "
            Else
                json = json & """initialInstallationDate"": """", "
            End If
            
            json = json & """initialSignageRequestNumber"": """", "
            
            If Not IsNull(rs.Fields("Construction Completion Date").Value) And IsDate(rs.Fields("Construction Completion Date").Value) Then
                json = json & """initialCompletionDate"": """ & Format(rs.Fields("Construction Completion Date").Value, "yyyy-mm-dd") & """, "
            Else
                json = json & """initialCompletionDate"": """", "
            End If
            
            json = json & """initialInstallationContractNumber"": """ & rs.Fields("Installation Contract Num").Value & """, "
            
            ' Check if all removalItems are empty
            Dim isEmptyRemovalItems As Boolean
            isEmptyRemovalItems = True ' Assume it's empty unless proven otherwise
            
            For i = LBound(removeFields) To UBound(removeFields)
                If Not IsNull(rs.Fields(removeFields(i))) And Len(rs.Fields(removeFields(i)).Value) > 0 Then
                    isEmptyRemovalItems = False ' Found a non-empty field
                    Exit For
                End If
            Next i
            
            ' If removalItems is empty, set it to an empty array, otherwise proceed with the normal logic
            If isEmptyRemovalItems Then
                json = json & """removalItems"": [], "
            Else
                json = json & """removalItems"":" & "[{"
                json = json & """removalLocation"": """", "
                json = json & """removalType"": """", "
                For i = LBound(removeFields) To UBound(removeFields)
                    fieldName = removeFields(i)
                   
                    If Not IsNull(rs.Fields(fieldName)) Then
                        If IsDate(rs.Fields(fieldName).Value) Then
                            json = json & """" & jsonRemoveFields(i) & """: """ & Format(rs.Fields(fieldName).Value, "yyyy-mm-dd") & """"
                        Else
                            json = json & """" & jsonRemoveFields(i) & """: """ & Replace(rs.Fields(fieldName).Value, """", "\""") & """"
                        End If
                    Else
                        json = json & """" & jsonRemoveFields(i) & """: """""
                    End If
                    If i < UBound(removeFields) Then
                        json = json & ", "
                    End If
                Next i
                json = json & "}], "
            End If
            
            ' Check if all maintenanceItems are empty
            Dim isEmptyMaintenanceItems As Boolean
            isEmptyMaintenanceItems = True ' Assume it's empty unless proven otherwise
            
            ' Check the fields in maintenanceItems (you can adjust the fields if necessary)
            If Len(Nz(rs.Fields("Re-Installation Contract Num").Value, "")) > 0 Or _
               Len(Nz(rs.Fields("Re-Installation Date").Value, "")) > 0 Then
                isEmptyMaintenanceItems = False ' Found a non-empty field
            End If
            
            ' If maintenanceItems is empty, set it to an empty array, otherwise proceed with the normal logic
            If isEmptyMaintenanceItems Then
                json = json & """maintenanceItems"": [], "
            Else
                json = json & """maintenanceItems"":" & "[{"
                json = json & """maintenanceReason"": """", "
                json = json & """reinstallationContractNumber"": """ & rs.Fields("Re-Installation Contract Num").Value & """, "
                If Not IsNull(rs.Fields("Re-Installation Date").Value) And IsDate(rs.Fields("Re-Installation Date").Value) Then
                    json = json & """reinstallationDate"": """ & Format(rs.Fields("Re-Installation Date").Value, "yyyy-mm-dd") & """, "
                Else
                    json = json & """reinstallationDate"": """", "
                End If
                json = json & """reinstallationLocation"": """", "
                json = json & """reinstallationType"": """""
                json = json & "}], "
            End If
                
            json = json & """appendItems"": []" & ", "
                  
            json = json & """uploadDraftPlan"": []" & ", "
            json = json & """requestTrackingNumberDescription"": """", "
            json = json & """eligibility"": """", "
            json = json & """eligibilityReasons"": """", "
            json = json & """eligibilityOther"": """""
            
            If Not rs.EOF Then
                json = json & "}, "
            Else
                json = json & "}"
            End If
            
            fileStream.WriteText json, 1
            json = ""
            exportedCount = exportedCount + 1
        End If
        
        rs.MoveNext
    Loop
       
    ' Replace the last comma with the closing square bracket
    Dim currentPosition As Long
    currentPosition = fileStream.Position
    
    ' Go to the beginning of the stream and read the content
    fileStream.Position = 0
    Dim fileContent As String
    fileContent = fileStream.ReadText(-1) ' Read all content
    
    ' Replace the last comma with a closing square bracket
    fileContent = Left(fileContent, Len(fileContent) - 4) & "]" ' Remove last comma and add "]"
    
    ' Write back the updated content
    fileStream.Position = 0
    fileStream.SetEOS ' Clear the stream before writing
    fileStream.WriteText fileContent ' Write the corrected content back to the stream
    
    ' Save the contents of the stream to a file
    fileStream.SaveToFile jsonFile, 2 ' 2 means overwrite the file if it exists
        
    ' Clean up
    fileStream.Close
    Set fileStream = Nothing
       
    rs.Close
    Set rs = Nothing
    Set db = Nothing
       
    MsgBox "Export complete for table: " & TableName & vbCrLf & _
           "Exported records: " & exportedCount & vbCrLf & _
           "Ignored records (missing critical fields): " & skippedCount, vbInformation
End Sub

'--- Checks if a string exists within an array ---
Function contains(s As String, arr As Variant) As Boolean
    ' The Filter function returns a zero-based array containing only elements that match the string 's'
    ' UBound gives the upper bound of that array
    ' If UBound is greater than -1, it means 's' was found in 'arr'
    contains = (UBound(Filter(arr, s)) > -1)
End Function

'--- Cleans a string by removing common line break characters and special invisible characters ---
Function cleanString(input_val As String) As String
    ' Removes different variations of line breaks and tab characters
    cleanString = Replace(input_val, vbCrLf, "")
    cleanString = Replace(cleanString, vbCr, "")
    cleanString = Replace(cleanString, vbLf, "")
    cleanString = Replace(cleanString, Chr(9), "")     ' Tab character
    cleanString = Replace(cleanString, Chr(10), "")    ' Line feed
    cleanString = Replace(cleanString, Chr(13), "")    ' Carriage return
    cleanString = Replace(cleanString, Chr(0), "")     ' Null character
    cleanString = Replace(cleanString, ChrW(&HFFFD), "") ' Unicode replacement character
    cleanString = Replace(cleanString, """", "\""")    ' Escapes double quotes by adding backslash
End Function

'--- Standardizes address suffixes and directions to full forms ---
Function GetStandardizedAddress(ByVal inputAddress As String) As String
    Dim suffixMap As Object, directionMap As Object
    Dim parts() As String
    Dim lastWord As String, secondLastWord As String
    Dim suffixKey As String, directionKey As String
    Dim result As String

    ' Create dictionaries (similar to hash maps) to store known suffix and direction translations
    Set suffixMap = CreateObject("Scripting.Dictionary")
    Set directionMap = CreateObject("Scripting.Dictionary")

    '--- Define known abbreviations and their standardized full forms ---
    suffixMap.Add "av", "Avenue"
    suffixMap.Add "ave", "Avenue"
    suffixMap.Add "aved", "Avenue"
    suffixMap.Add "ave.", "Avenue"
    suffixMap.Add "ave e", "Avenue East"
    suffixMap.Add "ave w", "Avenue West"
    suffixMap.Add "blvd", "Boulevard"
    suffixMap.Add "blvd.", "Boulevard"
    suffixMap.Add "blvd w", "Boulevard West"
    suffixMap.Add "blvd e", "Boulevard East"
    suffixMap.Add "cir", "Circle"
    suffixMap.Add "crcl", "Circle"
    suffixMap.Add "cres", "Crescent"
    suffixMap.Add "crt", "Court"
    suffixMap.Add "dr", "Drive"
    suffixMap.Add "dr s", "Drive South"
    suffixMap.Add "gdn", "Gardens"
    suffixMap.Add "gdns", "Gardens"
    suffixMap.Add "gt", "Gate"
    suffixMap.Add "hill", "Hill"
    suffixMap.Add "lwn", "Lawn"
    suffixMap.Add "park ave", "Park Avenue"
    suffixMap.Add "pkwy", "Parkway"
    suffixMap.Add "pl", "Place"
    suffixMap.Add "rd", "Road"
    suffixMap.Add "rd.", "Road"
    suffixMap.Add "st", "Street"
    suffixMap.Add "st.", "Street"
    suffixMap.Add "st e", "Street East"
    suffixMap.Add "st w", "Street West"
    suffixMap.Add "st s", "Street South"
    suffixMap.Add "ter", "Terrace"
    suffixMap.Add "trl", "Trail"

    '--- Define direction abbreviations and their full names ---
    directionMap.Add "n", "North"
    directionMap.Add "s", "South"
    directionMap.Add "e", "East"
    directionMap.Add "w", "West"
    directionMap.Add "north", "North"
    directionMap.Add "south", "South"
    directionMap.Add "east", "East"
    directionMap.Add "west", "West"

    '--- Clean and normalize the input address ---
    inputAddress = Trim(inputAddress)                            ' Remove leading/trailing spaces
    inputAddress = RemoveSpecialCharacters(inputAddress)         ' Remove unusual symbols
    inputAddress = StrConv(inputAddress, vbProperCase)           ' Convert to proper case (e.g., Main Street)
    inputAddress = cleanString(inputAddress)                     ' Remove special characters

    '--- Skip processing if address already ends with a known full-form suffix ---
    If AlreadyStandardized(inputAddress) Then
        GetStandardizedAddress = inputAddress
        Exit Function
    End If

    '--- Break address into words ---
    parts = Split(inputAddress, " ")
    If UBound(parts) < 0 Then
        GetStandardizedAddress = inputAddress
        Exit Function
    End If

    '--- Get last and second last word for possible suffix/direction matching ---
    lastWord = LCase(parts(UBound(parts)))
    If UBound(parts) >= 1 Then
        secondLastWord = LCase(parts(UBound(parts) - 1))
    Else
        secondLastWord = ""
    End If

    '--- Match case: suffix + direction ---
    If suffixMap.exists(secondLastWord) And directionMap.exists(lastWord) Then
        parts(UBound(parts) - 1) = suffixMap(secondLastWord)
        parts(UBound(parts)) = directionMap(lastWord)
        result = Join(parts, " ")
        GetStandardizedAddress = result
        Exit Function
    End If

    '--- Match case: suffix only ---
    If suffixMap.exists(lastWord) Then
        parts(UBound(parts)) = suffixMap(lastWord)
        result = Join(parts, " ")
        GetStandardizedAddress = result
        Exit Function
    End If

    '--- No match found; return address as-is ---
    GetStandardizedAddress = inputAddress
End Function

'--- Removes special characters except letters, numbers, space, period, and apostrophe ---
Function RemoveSpecialCharacters(ByVal inputStr As String) As String
    Dim outputStr As String
    Dim i As Integer
    Dim charCode As Integer

    outputStr = ""
    For i = 1 To Len(inputStr)
        charCode = Asc(Mid(inputStr, i, 1)) ' Get ASCII value of character
        ' Keep only valid characters: digits, uppercase/lowercase letters, space, period, apostrophe
        If (charCode >= 48 And charCode <= 57) Or _
           (charCode >= 65 And charCode <= 90) Or _
           (charCode >= 97 And charCode <= 122) Or _
           charCode = 32 Or charCode = 46 Or charCode = 39 Then
            outputStr = outputStr & Mid(inputStr, i, 1)
        End If
    Next i
    RemoveSpecialCharacters = outputStr
End Function

'--- Checks whether an address already ends in a valid full-form suffix (e.g., "Street") ---
Function AlreadyStandardized(ByVal addr As String) As Boolean
    Dim correctSuffixes As Variant
    Dim j As Integer

    correctSuffixes = Array("Street", "Avenue", "Boulevard", "Road", "Drive", _
                            "Lane", "Court", "Place", "Parkway", "Terrace", _
                            "Square", "Circle", "Trail", "Alley", "Cove", "Crescent")

    For j = LBound(correctSuffixes) To UBound(correctSuffixes)
        ' Compare the end of the address with each known suffix
        If LCase(Right(addr, Len(correctSuffixes(j)))) = LCase(correctSuffixes(j)) Then
            AlreadyStandardized = True
            Exit Function
        End If
    Next j
    AlreadyStandardized = False
End Function

'--- Simple utility to remove hash symbols from a URL string ---
Function RemoveHash(url As String) As String
    RemoveHash = Replace(url, "#", "")
End Function





