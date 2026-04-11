if ((Get-Location).Path -like "*start_server") {
    Set-Location ..
}

uvicorn app.main:app --reload