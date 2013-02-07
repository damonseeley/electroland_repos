# Microsoft Developer Studio Project File - Name="rocklights" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=rocklights - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "rocklights.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "rocklights.mak" CFG="rocklights - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "rocklights - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "rocklights - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "rocklights - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /Yu"stdafx.h" /FD /c
# ADD CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /FD /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib KiNETLibSimple.lib Traxess.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "rocklights - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /Yu"stdafx.h" /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR /FD /GZ /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 Traxess.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib KiNETLibSimple.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "rocklights - Win32 Release"
# Name "rocklights - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\Ambient.cpp
# End Source File
# Begin Source File

SOURCE=.\AmbientA.cpp
# End Source File
# Begin Source File

SOURCE=.\AmbientPixel.cpp
# End Source File
# Begin Source File

SOURCE=.\AmbRedStick.cpp
# End Source File
# Begin Source File

SOURCE=.\AmbTargetFlash.cpp
# End Source File
# Begin Source File

SOURCE=.\AmbTron.cpp
# End Source File
# Begin Source File

SOURCE=.\AV1Square.cpp
# End Source File
# Begin Source File

SOURCE=.\AV1SquareBUS.cpp
# End Source File
# Begin Source File

SOURCE=.\AV1SquareKit.cpp
# End Source File
# Begin Source File

SOURCE=.\AV1SquarePulse.cpp
# End Source File
# Begin Source File

SOURCE=.\AV4Square.cpp
# End Source File
# Begin Source File

SOURCE=.\AV9Square.cpp
# End Source File
# Begin Source File

SOURCE=.\Avatar.cpp
# End Source File
# Begin Source File

SOURCE=.\AVHuge.cpp
# End Source File
# Begin Source File

SOURCE=.\AVPlusSign.cpp
# End Source File
# Begin Source File

SOURCE=.\AVSinglePixelDance.cpp
# End Source File
# Begin Source File

SOURCE=.\AVSmall.cpp
# End Source File
# Begin Source File

SOURCE=.\BasePixel.cpp
# End Source File
# Begin Source File

SOURCE=.\ColorChannel.cpp
# End Source File
# Begin Source File

SOURCE=.\ColorElement.cpp
# End Source File
# Begin Source File

SOURCE=.\ColorElementR.cpp
# End Source File
# Begin Source File

SOURCE=.\ColorElementRGB.cpp
# End Source File
# Begin Source File

SOURCE=.\DataEnabler.cpp
# End Source File
# Begin Source File

SOURCE=.\DataEnablerFile.cpp
# End Source File
# Begin Source File

SOURCE=.\DataEnablers.cpp
# End Source File
# Begin Source File

SOURCE=.\Dummies.cpp
# End Source File
# Begin Source File

SOURCE=.\globals.cpp
# End Source File
# Begin Source File

SOURCE=.\ICycle.cpp
# End Source File
# Begin Source File

SOURCE=.\IGeneric.cpp
# End Source File
# Begin Source File

SOURCE=.\IHoldAndFade.cpp
# End Source File
# Begin Source File

SOURCE=.\InterpGen.cpp
# End Source File
# Begin Source File

SOURCE=.\Interpolator.cpp
# End Source File
# Begin Source File

SOURCE=.\Interpolators.cpp
# End Source File
# Begin Source File

SOURCE=.\LECoveStick.cpp
# End Source File
# Begin Source File

SOURCE=.\LETargetCircle.cpp
# End Source File
# Begin Source File

SOURCE=.\LightElement.cpp
# End Source File
# Begin Source File

SOURCE=.\LightFile.cpp
# End Source File
# Begin Source File

SOURCE=.\MasterController.cpp
# End Source File
# Begin Source File

SOURCE=.\MCA.cpp
# End Source File
# Begin Source File

SOURCE=.\MCB.cpp
# End Source File
# Begin Source File

SOURCE=.\MCCrowded.cpp
# End Source File
# Begin Source File

SOURCE=.\MCEmpty.cpp
# End Source File
# Begin Source File

SOURCE=.\MCH.cpp
# End Source File
# Begin Source File

SOURCE=.\MCMobbed.cpp
# End Source File
# Begin Source File

SOURCE=.\MCPopulated.cpp
# End Source File
# Begin Source File

SOURCE=.\MCSinglePixelDance.cpp
# End Source File
# Begin Source File

SOURCE=.\MCSparce.cpp
# End Source File
# Begin Source File

SOURCE=.\MCTargetFlash.cpp
# End Source File
# Begin Source File

SOURCE=.\OffsetPixel.cpp
# End Source File
# Begin Source File

SOURCE=.\Panel.cpp
# End Source File
# Begin Source File

SOURCE=.\Panels.cpp
# End Source File
# Begin Source File

SOURCE=.\Pattern.cpp
# End Source File
# Begin Source File

SOURCE=.\Pattern1Square.cpp
# End Source File
# Begin Source File

SOURCE=.\Pattern9Square.cpp
# End Source File
# Begin Source File

SOURCE=.\PatternA.cpp
# End Source File
# Begin Source File

SOURCE=.\PatternB.cpp
# End Source File
# Begin Source File

SOURCE=.\PatternPixelDance.cpp
# End Source File
# Begin Source File

SOURCE=.\PatternPlusSign.cpp
# End Source File
# Begin Source File

SOURCE=.\PeopleStats.cpp
# End Source File
# Begin Source File

SOURCE=.\PersonStats.cpp
# End Source File
# Begin Source File

SOURCE=.\Pixel.cpp
# End Source File
# Begin Source File

SOURCE=.\profile.cpp
# End Source File
# Begin Source File

SOURCE=.\rocklights.cpp
# End Source File
# Begin Source File

SOURCE=.\SubPixel.cpp
# End Source File
# Begin Source File

SOURCE=.\TargetPixel.cpp
# End Source File
# Begin Source File

SOURCE=.\Tracker.cpp
# End Source File
# Begin Source File

SOURCE=.\WorldStats.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\Ambient.h
# End Source File
# Begin Source File

SOURCE=.\AmbientA.h
# End Source File
# Begin Source File

SOURCE=.\AmbientPixel.h
# End Source File
# Begin Source File

SOURCE=.\AmbRedStick.h
# End Source File
# Begin Source File

SOURCE=.\AmbTargetFlash.h
# End Source File
# Begin Source File

SOURCE=.\AmbTron.h
# End Source File
# Begin Source File

SOURCE=.\AV1Square.h
# End Source File
# Begin Source File

SOURCE=.\AV1SquareBUS.h
# End Source File
# Begin Source File

SOURCE=.\AV1SquareKit.h
# End Source File
# Begin Source File

SOURCE=.\AV1SquarePulse.h
# End Source File
# Begin Source File

SOURCE=.\AV4Square.h
# End Source File
# Begin Source File

SOURCE=.\AV9Square.h
# End Source File
# Begin Source File

SOURCE=.\Avatar.h
# End Source File
# Begin Source File

SOURCE=.\AVHuge.h
# End Source File
# Begin Source File

SOURCE=.\AVPlusSign.h
# End Source File
# Begin Source File

SOURCE=.\AVSinglePixelDance.h
# End Source File
# Begin Source File

SOURCE=.\AVSmall.h
# End Source File
# Begin Source File

SOURCE=.\BasePixel.h
# End Source File
# Begin Source File

SOURCE=.\ColorChannel.h
# End Source File
# Begin Source File

SOURCE=.\ColorElement.h
# End Source File
# Begin Source File

SOURCE=.\ColorElementR.h
# End Source File
# Begin Source File

SOURCE=.\ColorElementRGB.h
# End Source File
# Begin Source File

SOURCE=.\DataEnabler.h
# End Source File
# Begin Source File

SOURCE=.\DataEnablerFile.h
# End Source File
# Begin Source File

SOURCE=.\DataEnablers.h
# End Source File
# Begin Source File

SOURCE=.\debug.h
# End Source File
# Begin Source File

SOURCE=.\Dummies.h
# End Source File
# Begin Source File

SOURCE=.\globals.h
# End Source File
# Begin Source File

SOURCE=.\ICycle.h
# End Source File
# Begin Source File

SOURCE=.\IGeneric.h
# End Source File
# Begin Source File

SOURCE=.\IHoldAndFade.h
# End Source File
# Begin Source File

SOURCE=.\InterpGen.h
# End Source File
# Begin Source File

SOURCE=.\Interpolator.h
# End Source File
# Begin Source File

SOURCE=.\Interpolators.h
# End Source File
# Begin Source File

SOURCE=.\LECoveStick.h
# End Source File
# Begin Source File

SOURCE=.\LETargetCircle.h
# End Source File
# Begin Source File

SOURCE=.\LightElement.h
# End Source File
# Begin Source File

SOURCE=.\LightFile.h
# End Source File
# Begin Source File

SOURCE=.\MasterController.h
# End Source File
# Begin Source File

SOURCE=.\MCA.h
# End Source File
# Begin Source File

SOURCE=.\MCB.h
# End Source File
# Begin Source File

SOURCE=.\MCCrowded.h
# End Source File
# Begin Source File

SOURCE=.\MCEmpty.h
# End Source File
# Begin Source File

SOURCE=.\MCMobbed.h
# End Source File
# Begin Source File

SOURCE=.\MCPopulated.h
# End Source File
# Begin Source File

SOURCE=.\MCSinglePixelDance.h
# End Source File
# Begin Source File

SOURCE=.\MCSparce.h
# End Source File
# Begin Source File

SOURCE=.\MCTargetFlash.h
# End Source File
# Begin Source File

SOURCE=.\OffsetPixel.h
# End Source File
# Begin Source File

SOURCE=.\Panel.h
# End Source File
# Begin Source File

SOURCE=.\Panels.h
# End Source File
# Begin Source File

SOURCE=.\Pattern.h
# End Source File
# Begin Source File

SOURCE=.\Pattern1Square.h
# End Source File
# Begin Source File

SOURCE=.\Pattern9Square.h
# End Source File
# Begin Source File

SOURCE=.\PatternA.h
# End Source File
# Begin Source File

SOURCE=.\PatternB.h
# End Source File
# Begin Source File

SOURCE=.\PatternPixelDance.h
# End Source File
# Begin Source File

SOURCE=.\PatternPlusSign.h
# End Source File
# Begin Source File

SOURCE=.\PeopleStats.h
# End Source File
# Begin Source File

SOURCE=.\PersonStats.h
# End Source File
# Begin Source File

SOURCE=.\Pixel.h
# End Source File
# Begin Source File

SOURCE=.\profile.h
# End Source File
# Begin Source File

SOURCE=.\SubPixel.h
# End Source File
# Begin Source File

SOURCE=.\TargetPixel.h
# End Source File
# Begin Source File

SOURCE=.\Tracker.h
# End Source File
# Begin Source File

SOURCE=.\WorldStats.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# Begin Source File

SOURCE=.\DataEnablers.txt
# End Source File
# Begin Source File

SOURCE=.\lights.txt
# End Source File
# Begin Source File

SOURCE=.\profile.txt
# End Source File
# Begin Source File

SOURCE=.\todo.txt
# End Source File
# End Target
# End Project
