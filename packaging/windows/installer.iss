; Inno Setup script: builds Proxmino-Setup-x64.exe from the PyInstaller output in dist\Proxmino.
; Installs per user (no admin prompt). Built by .github/workflows/release.yml.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{3F0B6E9A-2C4D-4E8B-A1F7-6D2C9B8E5A13}
AppName=Proxmino
AppVersion={#AppVersion}
AppPublisher=jukeiss
AppPublisherURL=https://github.com/jukeiss/proxmino
DefaultDirName={localappdata}\Programs\Proxmino
DefaultGroupName=Proxmino
PrivilegesRequired=lowest
OutputDir=..\..\out
OutputBaseFilename=Proxmino-Setup-x64
SetupIconFile=..\..\build\icon.ico
UninstallDisplayIcon={app}\Proxmino.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\Proxmino\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Proxmino"; Filename: "{app}\Proxmino.exe"
Name: "{group}\Uninstall Proxmino"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Proxmino"; Filename: "{app}\Proxmino.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Proxmino.exe"; Description: "Start Proxmino now"; Flags: nowait postinstall skipifsilent
