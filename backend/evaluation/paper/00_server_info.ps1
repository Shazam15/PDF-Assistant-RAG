# Tabla 2 (perfil lan_client): describe el SERVIDOR de inferencia Windows
# (el equipo con Ollama y la Tesla T4). No necesita Python ni el repositorio.
#
#   powershell -ExecutionPolicy Bypass -File 00_server_info.ps1 -Output server_info.json
#
# Copia server_info.json al cliente ATLAS y pasalo a 00_environment.py --server-info.
param([string]$Output = "server_info.json")

$cpu = Get-CimInstance Win32_Processor
$system = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem

$gpus = @()
$cudaDriver = $null
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $rows = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    foreach ($row in $rows) {
        $parts = $row.Split(",") | ForEach-Object { $_.Trim() }
        $gpus += [ordered]@{ name = $parts[0]; memory_gb = [math]::Round([double]$parts[1] / 1024, 1); driver = $parts[2] }
    }
    $header = (& nvidia-smi) -join "`n"
    if ($header -match "CUDA Version:\s*([\d.]+)") { $cudaDriver = $Matches[1] }
}

$ollamaVersion = $null
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) { $ollamaVersion = (& ollama --version 2>&1 | Out-String).Trim() }

# Variables que cambian el rendimiento de Ollama y deben declararse.
$ollamaEnv = @()
foreach ($name in "OLLAMA_KEEP_ALIVE", "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS",
                  "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_CONTEXT_LENGTH") {
    $value = [Environment]::GetEnvironmentVariable($name, "Machine")
    if (-not $value) { $value = [Environment]::GetEnvironmentVariable($name, "User") }
    if ($value) { $ollamaEnv += "$name=$value" }
}

$info = [ordered]@{
    captured_at        = (Get-Date).ToUniversalTime().ToString("o")
    cpu                = "$(($cpu | Select-Object -First 1).Name.Trim()) - $(@($cpu).Count) socket(s), " +
                         "$(($cpu | Measure-Object NumberOfCores -Sum).Sum) nucleos / " +
                         "$(($cpu | Measure-Object NumberOfLogicalProcessors -Sum).Sum) hilos"
    ram_gb             = [math]::Round($system.TotalPhysicalMemory / 1GB, 1)
    gpus               = $gpus
    cuda_driver_version = $cudaDriver
    os                 = "$($os.Caption) $($os.Version)"
    ollama_version     = $ollamaVersion
    ollama_env         = ($ollamaEnv -join ", ")
}
$info | ConvertTo-Json -Depth 4 | Set-Content -Path $Output -Encoding UTF8
Write-Output "Guardado en $Output"
$info | ConvertTo-Json -Depth 4
