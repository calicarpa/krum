@echo off

if "%1" == "html" (
	sphinx-build -b html . _build\html
	goto :end
)

if "%1" == "clean" (
	rmdir /s /q _build
	goto :end
)

echo Available targets: html, clean
:end