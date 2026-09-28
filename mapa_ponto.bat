@echo off
rem Mapas em volta de um ponto (150 km para cada lado), separados dos mapas diarios.
rem   Dois cliques: pergunta o nome do ponto e a coordenada.
rem   Os mapas saem na pasta previsao_pontos\<nome do ponto>.
setlocal
cd /d "%~dp0"

set "PY="
for %%V in (3.13 3.12 3.11) do (
  if not defined PY py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
)
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python nao encontrado. Instale o Python 3.13 em python.org e marque "Add python.exe to PATH".
  pause
  exit /b 1
)

%PY% rodar_local.py --ponto %*
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (echo Pronto. Os mapas estao na pasta previsao_pontos.) else (echo Terminou com erro. Veja o log mais recente na pasta logs.)
pause
exit /b %RC%
