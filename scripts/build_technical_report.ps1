param(
    [switch]$SkipFigures
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv-amd\Scripts\python.exe"

if (-not (Test-Path $python)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $python = $pythonCommand.Source
}

$pdfLatex = (Get-Command pdflatex -ErrorAction Stop).Source

Push-Location $root
try {
    if (-not $SkipFigures) {
        & $python "scripts\generate_report_figures.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Figure generation failed with exit code $LASTEXITCODE"
        }
    }

    Push-Location "docs"
    try {
        foreach ($pass in 1..2) {
            & $pdfLatex -interaction=nonstopmode -halt-on-error "technical_report.tex"
            if ($LASTEXITCODE -ne 0) {
                throw "LaTeX pass $pass failed with exit code $LASTEXITCODE"
            }
        }
    }
    finally {
        Pop-Location
    }

    Remove-Item "docs\technical_report.aux", "docs\technical_report.log", "docs\technical_report.out" `
        -ErrorAction SilentlyContinue
    Write-Output "Built docs\technical_report.pdf"
}
finally {
    Pop-Location
}