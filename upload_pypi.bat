@echo off
REM OllamaAid - Build and upload to PyPI
setlocal
cd /d "%~dp0"

if not defined PYTHON set "PYTHON=python"
set "VERSION_FILE=ollama_aid\__version__.py"

echo === OllamaAid PyPI Upload ===

echo [1/6] Bumping patch version...
%PYTHON% -c "import re,sys;p='%VERSION_FILE%'.replace('\\','/');t=open(p).read();m=re.search(r'(__version__\s*=\s*\"(\d+\.\d+\.)(\d+)\")',t);old=m.group(2)+m.group(3);new=m.group(2)+str(int(m.group(3))+1);open(p,'w').write(t.replace(m.group(1),'__version__ = \"'+new+'\"'));print(f'  {old} -> {new}')"
if %errorlevel% neq 0 (echo Version bump failed! & exit /b 1)

echo [2/6] Cleaning old builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

echo [3/6] Installing build tools...
%PYTHON% -m pip install --upgrade build twine -q

echo [4/6] Building package...
%PYTHON% -m build
if %errorlevel% neq 0 (echo Build failed! & exit /b 1)
%PYTHON% -m twine check dist\*
if %errorlevel% neq 0 (echo Check failed! & exit /b 1)

echo [5/6] Uploading to PyPI...
%PYTHON% -m twine upload dist\*
if %errorlevel% neq 0 (echo Upload failed! & exit /b 1)

echo [6/6] Committing and pushing to GitHub...
for /f "delims=" %%v in ('%PYTHON% -c "import re;t=open('%VERSION_FILE%'.replace(chr(92),'/')).read();m=re.search(r'__version__\s*=\s*\"([\d.]+)\"',t);print(m.group(1))"') do set "NEW_VER=%%v"
git add -A
git commit -m "Release v%NEW_VER%"
git tag "v%NEW_VER%"
git push origin main --tags
if %errorlevel% neq 0 (echo Git push failed! & exit /b 1)

echo === Done! Released v%NEW_VER% ===
endlocal
