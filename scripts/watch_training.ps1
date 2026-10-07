$ErrorActionPreference = 'Stop'
$project = Join-Path (Split-Path $PSScriptRoot -Parent) 'synthetic_vs_real_cv'
while ($true) {
    $active = Get-Content (Join-Path $project 'configs/active_dataset.json') -Raw | ConvertFrom-Json
    $statusPath = Join-Path $project "results/runs/$($active.run)/queue_status.json"
    if (Test-Path $statusPath) {
        try { $s = Get-Content $statusPath -Raw | ConvertFrom-Json } catch { Start-Sleep -Seconds 1; continue }
        Clear-Host
        Write-Host "Queue: $($active.run)"
        Write-Host "Model: $($s.current)"
        Write-Host "Status: $($s.status)"
        Write-Host "Epoch: $($s.epoch) / $($s.total_epochs)"
        Write-Host "Completed models: $($s.completed.Count) / $($s.config.experiments.Count)"
        Write-Host "Updated: $($s.updated)"
        if ($s.total_epochs -gt 0) { Write-Progress -Activity $s.current -Status $s.status -PercentComplete ([Math]::Min(100,100*$s.epoch/$s.total_epochs)) }
        if ($s.error) { Write-Host $s.error -ForegroundColor Red }
        if ($s.status -in @('complete','failed','superseded')) { break }
    } else { Write-Host 'Waiting for queue to start...' }
    Start-Sleep -Seconds 5
}
