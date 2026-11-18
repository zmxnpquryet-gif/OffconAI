; OffconAI Desktop — Windows Installer Wizard (Inno Setup 6)

#ifndef AppVersion
  #define AppVersion "2.3.0"
#endif
#ifndef RepoRoot
  #define RepoRoot "..\..\.."
#endif
#ifndef DistDir
  #define DistDir "..\..\build\install\desktop\bin"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\build\installer"
#endif

[Setup]
AppId={{B8E2D7A4-9C41-4E9B-8A12-3F7E4C9A2105}
AppName=OffconAI Desktop
AppVersion={#AppVersion}
AppVerName=OffconAI Desktop {#AppVersion}
AppPublisher=OffconAI
AppPublisherURL=https://zmxnpquryet-gif.github.io/OffconAI/
AppSupportURL=https://github.com/zmxnpquryet-gif/OffconAI
AppUpdatesURL=https://github.com/zmxnpquryet-gif/OffconAI/releases
DefaultDirName={autopf}\OffconAI Desktop
DefaultGroupName=OffconAI Desktop
DisableProgramGroupPage=yes
LicenseFile={#RepoRoot}\LICENSE
SetupIconFile={#RepoRoot}\desktop\assets\offcon.ico
UninstallDisplayIcon={app}\offcon-desktop.exe
UninstallDisplayName=OffconAI Desktop
OutputDir={#OutputDir}
OutputBaseFilename=OffconAI-desktop-windows-x86_64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
CloseApplicationsFilter=offcon-desktop.exe,llama-server.exe
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#DistDir}\offcon-desktop.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#DistDir}\llama-server.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\desktop\assets\offcon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\OffconAI Desktop"; Filename: "{app}\offcon-desktop.exe"; IconFilename: "{app}\offcon.ico"
Name: "{autoprograms}\Uninstall OffconAI Desktop"; Filename: "{uninstallexe}"
Name: "{autodesktop}\OffconAI Desktop"; Filename: "{app}\offcon-desktop.exe"; IconFilename: "{app}\offcon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\offcon-desktop.exe"; Description: "{cm:LaunchProgram,OffconAI Desktop}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: string;
begin
  if (CurUninstallStep = usPostUninstall) and (not UninstallSilent) then
  begin
    DataDir := ExpandConstant('{localappdata}\OffconAI');
    if DirExists(DataDir) then
    begin
      if MsgBox(
        'Do you also want to delete downloaded GGUF models, logs, and settings in:' + #13#10 + DataDir + #13#10#13#10 + 'Choose No to keep your downloaded models.',
        mbConfirmation,
        MB_YESNO or MB_DEFBUTTON2
      ) = IDYES then
      begin
        DelTree(DataDir, True, True, True);
      end;
    end;
  end;
end;
