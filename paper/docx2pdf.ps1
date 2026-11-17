param([Parameter(Mandatory=$true)][string]$In, [Parameter(Mandatory=$true)][string]$Out)
# Convert a .docx to PDF with Microsoft Word (COM automation).
$In = [System.IO.Path]::GetFullPath($In)
$Out = [System.IO.Path]::GetFullPath($Out)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open((Resolve-Path $In).Path, $false, $true)
    $doc.Fields.Update() | Out-Null
    $doc.SaveAs([ref]$Out, [ref]17)   # 17 = wdFormatPDF
    $doc.Close([ref]0)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
Write-Output "saved $Out"
