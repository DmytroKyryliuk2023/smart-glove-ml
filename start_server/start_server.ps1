if ((Get-Location).Path -notlike "*start_server") {
    Set-Location start_server
}

uvicorn app.main:app --reload