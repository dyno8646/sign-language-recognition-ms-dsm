param(
    [int]$Port = 8000
)

$env:APP_PORT = "$Port"
uvicorn app.main:app --reload --host 0.0.0.0 --port $Port
