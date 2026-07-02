@echo off
setlocal

:: Check if required arguments are provided
if "%~1"=="" (
    echo Usage: %0 "prompt" "output_file.png"
    exit /b 1
)

if "%~2"=="" (
    echo Usage: %0 "prompt" "output_file.png"
    exit /b 1
)

:: Set variables
set "prompt=%~1"
set "output_file=%~2"

:: Create temporary JSON file
set "temp_json=%TEMP%\image_request.json"
echo {"prompt": "%prompt%", "stream": false} > "%temp_json%"

:: Send request to image server and capture response
echo Generating image for prompt: %prompt%
echo.

:: Use curl to send POST request and capture response
curl.exe -X POST http://127.0.0.1:8001/v1/image/generations ^
-H "Content-Type: application/json" ^
-d @%temp_json% ^
-o "%TEMP%\response.json"

:: Check if response was received
if exist "%TEMP%\response.json" (
    :: Extract base64 image data
    powershell -Command ^
    "$json = Get-Content '%TEMP%\response.json' | ConvertFrom-Json; " ^
    "[System.IO.File]::WriteAllBytes('%output_file%', [System.Convert]::FromBase64String($json.image))"

    :: Check if image file was created successfully
    if exist "%output_file%" (
        echo Image saved to: %output_file%
    ) else (
        echo Error: Failed to save image.
    )
) else (
    echo Error: Failed to receive response.
)

:: Clean up temp files
del "%temp_json%" 2>nul
del "%TEMP%\response.json" 2>nul

endlocal