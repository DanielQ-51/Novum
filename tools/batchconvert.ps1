# Define the range
$start = 0
$end = 249

for ($i = $start; $i -le $end; $i++) {
    # Formats number to 0000
    $num = $i.ToString("0000")
    
    $inputPath  = "..\assets\vdb\industrial\industrial_chimney_smoke\industrial_chimney_smoke_VDB\industrial_chimney_smoke_$num.vdb"
    $outputPath = "..\assets\vdb\nvdb\industrial\smoke_$num.nvdb"

    Write-Host "Converting frame $num..." -ForegroundColor Cyan
    
    # Run the converter
    .\nanovdb_convert.exe "$inputPath" "$outputPath"
}

Write-Host "Batch Conversion Complete!" -ForegroundColor Green