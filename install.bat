@echo off
setlocal

REM Choose env name (change if you want)
set ENV_NAME=cxr-demo

echo Creating conda env %ENV_NAME% with Python 3.10...
conda create -n %ENV_NAME% python=3.10 -y || goto :error

call conda activate %ENV_NAME%

echo Installing base requirements...
pip install -r requirements.txt || goto :error

echo Installing Keras 3 (no deps) to avoid TF 2.15 resolver conflict...
pip install --no-deps keras==3.3.3 || goto :error

echo Done.
echo To run: 
echo   conda activate %ENV_NAME%
echo   streamlit run app/streamlit_app.py
exit /b 0

:error
echo.
echo Installation failed. See the error above.
exit /b 1
