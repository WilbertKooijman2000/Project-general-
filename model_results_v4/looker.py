cd "C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)"
Get-ChildItem -Recurse -Include *.csv,*.json,*.parquet,*.pkl | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    if ($_.Extension -eq ".csv") {
        $rows = (Get-Content $_.FullName | Measure-Object -Line).Lines
        Write-Host "$($_.FullName)  |  $rows rows  |  $size MB"
    } else {
        Write-Host "$($_.FullName)  |  ?  |  $size MB"
    }
}