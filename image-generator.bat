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

:: Send request to image server
echo Generating image for prompt: %prompt%
echo.

:: Use curl to send POST request
curl.exe -X POST http://127.0.0.1:8001/v1/image/generations ^
-H "Content-Type: application/json" ^
-d @%temp_json% ^
-o "%output_file%"

:: Check if file was created
if exist "%output_file%" (
    echo Image saved to: %output_file%
) else (
    echo Error: Failed to save image.
)

:: Clean up temp file
del "%temp_json%" 2>nul

endlocal
