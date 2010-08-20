CoDeSys+С                     @        @   2.3.10.0    @?    @                                     p`L +    @      HH               М>L          @   n   C:\TWINCAT\PLC\LIB\TcpIp.lib @                                                                                          F_GETVERSIONTCPIP               nVersionElement           §џ                 F_GetVersionTcpIp                                     Ќ{D      џџџџ           FB_SOCKETACCEPT        
   fbAdsRdWrt                            ADSRDWRT    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              response                ST_TcIpConnSvrResponse    §џ              request                ST_SockAddr    §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system	   hListener              	   T_HSOCKET   §џ       x    Listener handle identifying a socket that has been placed in a listening state with the FB_SocketListen function block    bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ       3    States the time before the function is cancelled.    	   bAccepted            §џ
       ;    TRUE = new connection is made. FALSE = no new connection.    bBusy            §џ              bError            §џ              nErrId           §џ              hSocket              	   T_HSOCKET   §џ       V   This returned value is a handle for the socket on which the actual connection is made.            Ќ{D      џџџџ           FB_SOCKETCLOSE        
   fbAdsWrite                          ADSWRITE    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system   hSocket              	   T_HSOCKET   §џ       4    Local or remote client or listener socket to close.   bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ                       Ќ{D      џџџџ           FB_SOCKETCLOSEALL        
   fbAdsWrite                          ADSWRITE    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system   bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ       3    States the time before the function is cancelled.       bBusy            §џ	              bError            §џ
              nErrId           §џ                       Ќ{D      џџџџ           FB_SOCKETCONNECT        
   fbAdsRdWrt                            ADSRDWRT    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              request                ST_SockAddr    §џ              response                ST_TcIpConnSvrResponse    §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system   sRemoteHost               §џ       X    Remote (server) address. String containing an (Ipv4) Internet Protocol dotted address.    nRemotePort           §џ       .    Remote (server) Internet Protocol (IP) port.    bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout    ШЏ     §џ       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ              hSocket              	   T_HSOCKET   §џ       V   This returned value is a handle for the socket on which the actual connection is made.            Ќ{D      џџџџ           FB_SOCKETLISTEN        
   fbAdsRdWrt                            ADSRDWRT    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              request                ST_SockAddr    §џ              response                ST_TcIpConnSvrResponse    §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system
   sLocalHost               §џ       W    Local (server) address. String containing an (Ipv4) Internet Protocol dotted address. 
   nLocalPort           §џ       -    Local (server) Internet Protocol (IP) port.    bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ	       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ           	   hListener              	   T_HSOCKET   §џ       _   This returned value is a handle for the listener socket on which the actual connection is made.            Ќ{D      џџџџ           FB_SOCKETRECEIVE           fbAdsReadEx                        	   ADSREADEX    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system   hSocket              	   T_HSOCKET   §џ       ?    Handle for the socket on which the actual connection is made.    cbLen           §џ       3    Contains the max. number of bytes to be received.    pDest           §џ       ;    Contains the address of the buffer for the received data.    bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ	       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ           	   nRecBytes           §џ       2    Contains the number of bytes currently received.             Ќ{D      џџџџ           FB_SOCKETSEND        
   fbAdsWrite                          ADSWRITE    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system   hSocket              	   T_HSOCKET   §џ       ?    Handle for the socket on which the actual connection is made.    cbLen           §џ       *    Contains the number of bytes to be send.    pSrc           §џ       D    Contains the address of the buffer containing the data to be send.    bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ	       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ                       Ќ{D      џџџџ           FB_SOCKETUDPCREATE        
   fbAdsRdWrt                            ADSRDWRT    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              request                ST_SockAddr    §џ              response                ST_TcIpConnSvrResponse    §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system
   sLocalHost               §џ       N    Local address. String containing an (Ipv4) Internet Protocol dotted address. 
   nLocalPort           §џ	       $    Local Internet Protocol (IP) port.    bExecute            §џ
       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ              hSocket              	   T_HSOCKET   §џ       ?   This returned value is a handle for the bind (reserved) socket.            Ќ{D      џџџџ           FB_SOCKETUDPRECEIVEFROM           fbAdsReadEx                        	   ADSREADEX    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              buffer                ST_TcIpConnSvrUdpBuffer    §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system   hSocket              	   T_HSOCKET   §џ       ?    Handle for the socket on which the actual connection is made.    cbLen           §џ       3    Contains the max. number of bytes to be received.    pDest           §џ       ;    Contains the address of the buffer for the received data.    bExecute            §џ       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ	       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ              sRemoteHost               §џ       p    Remote address from which the data was received. String containing an (Ipv4) Internet Protocol dotted address.    nRemotePort           §џ       G    Remote Internet Protocol (IP) port  from which the data was received. 	   nRecBytes           §џ       2    Contains the number of bytes currently received.             Ќ{D      џџџџ           FB_SOCKETUDPSENDTO        
   fbAdsWrite                          ADSWRITE    §џ              fbRTrig                 R_TRIG    §џ              nStep            §џ              buffer                ST_TcIpConnSvrUdpBuffer    §џ              	   sSrvNetId           ''    
   T_AmsNetId   §џ       Y    The network address of the TcpIpserver.exe. If empty string=>server runs on local system   hSocket              	   T_HSOCKET   §џ       ?    Handle for the socket on which the actual connection is made.    sRemoteHost               §џ       d    Remote address of the target socket. String containing an (Ipv4) Internet Protocol dotted address.    nRemotePort           §џ       :    Remote Internet Protocol (IP) port of the target socket.    cbLen           §џ       *    Contains the number of bytes to be send.    pSrc           §џ	       D    Contains the address of the buffer containing the data to be send.    bExecute            §џ
       F    Function block execution is triggered by a rising edge at this input.   tTimeout         §џ       3    States the time before the function is cancelled.       bBusy            §џ              bError            §џ              nErrId           §џ                       Ќ{D      џџџџ    q   C:\TWINCAT\PLC\LIB\TcSystem.lib @                                                                                L          ADSCLEAREVENTS           fbAdsClearEvents                            FW_AdsClearEvents    §џ                 NetID            
   T_AmsNetId   §џ              bClear            §џ              iMode           §џ              tTimeout         §џ                 bBusy            §џ	              bErr            §џ
              iErrId           §џ                       ЊK     џџџџ        
   ADSLOGDINT               msgCtrlMask           §џ           	   msgFmtStr               T_MaxString   §џ              dintArg           §џ              
   ADSLOGDINT                                     ЊK      џџџџ           ADSLOGEVENT           fbAdsLogEvent                                               FW_AdsLogEvent    §џ           	      NETID            
   T_AmsNetId   §џ              PORT           §џ              Event            §џ           	   EventQuit            §џ              EventConfigData               TcEvent   §џ              EventDataAddress           §џ       	    pointer    EventDataLength           §џ	           	   FbCleanup            §џ
              TMOUT         §џ              
   EventState           §џ              Err            §џ              ErrId           §џ              Quit            §џ                       ЊK     џџџџ           ADSLOGLREAL               msgCtrlMask           §џ           	   msgFmtStr               T_MaxString   §џ              lrealArg                        §џ                 ADSLOGLREAL                                     ЊK      џџџџ        	   ADSLOGSTR               msgCtrlMask           §џ           	   msgFmtStr               T_MaxString   §џ              strArg               T_MaxString   §џ              	   ADSLOGSTR                                     ЊK      џџџџ           ADSRDDEVINFO           fbAdsReadDeviceInfo                              FW_AdsReadDeviceInfo    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              RDINFO            §џ              TMOUT         §џ                 BUSY            §џ	              ERR            §џ
              ERRID           §џ              DEVNAME               §џ              DEVVER           §џ                       ЊK     џџџџ        
   ADSRDSTATE           fbAdsReadState                              FW_AdsReadState    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              RDSTATE            §џ              TMOUT         §џ                 BUSY            §џ	              ERR            §џ
              ERRID           §џ              ADSSTATE           §џ              DEVSTATE           §џ                       ЊK     џџџџ           ADSRDWRT        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ           
      NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              IDXGRP           §џ              IDXOFFS           §џ              WRITELEN           §џ              READLEN           §џ              SRCADDR           §џ	              DESTADDR           §џ
              WRTRD            §џ              TMOUT         §џ                 BUSY            §џ              ERR            §џ              ERRID           §џ                       ЊK     џџџџ        
   ADSRDWRTEX        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ           
      NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              IDXGRP           §џ              IDXOFFS           §џ              WRITELEN           §џ              READLEN           §џ              SRCADDR           §џ	              DESTADDR           §џ
              WRTRD            §џ              TMOUT         §џ                 BUSY            §џ              ERR            §џ              ERRID           §џ              COUNT_R           §џ           count of bytes actually read             ЊK     џџџџ           ADSRDWRTIND           fbAdsRdWrtInd                         FW_AdsRdWrtInd    §џ                 CLEAR            §џ           	      VALID            §џ              NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              INVOKEID           §џ	              IDXGRP           §џ
              IDXOFFS           §џ              RDLENGTH           §џ           	   WRTLENGTH           §џ              DATAADDR           §џ                       ЊK      џџџџ           ADSRDWRTRES           fbAdsRdWrtRes                      FW_AdsRdWrtRes    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              INVOKEID           §џ              RESULT           §џ              LEN           §џ              DATAADDR           §џ              RESPOND            §џ	                           ЊK      џџџџ           ADSREAD        	   fbAdsRead                              
   FW_AdsRead    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              IDXGRP           §џ              IDXOFFS           §џ              LEN           §џ              DESTADDR           §џ              READ            §џ	              TMOUT         §џ
                 BUSY            §џ              ERR            §џ              ERRID           §џ                       ЊK     џџџџ        	   ADSREADEX        	   fbAdsRead                              
   FW_AdsRead    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              IDXGRP           §џ              IDXOFFS           §џ              LEN           §џ              DESTADDR           §џ              READ            §џ	              TMOUT         §џ
                 BUSY            §џ              ERR            §џ              ERRID           §џ              COUNT_R           §џ           count of bytes actually read             ЊK     џџџџ        
   ADSREADIND           fbAdsReadInd        	               FW_AdsReadInd    §џ                 CLEAR            §џ                 VALID            §џ              NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              INVOKEID           §џ	              IDXGRP           §џ
              IDXOFFS           §џ              LENGTH           §џ                       ЊK      џџџџ        
   ADSREADRES           fbAdsReadRes                      FW_AdsReadRes    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              INVOKEID           §џ              RESULT           §џ              LEN           §џ              DATAADDR           §џ              RESPOND            §џ	                           ЊK      џџџџ           ADSWRITE        
   fbAdsWrite                                FW_AdsWrite    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              IDXGRP           §џ              IDXOFFS           §џ              LEN           §џ              SRCADDR           §џ              WRITE            §џ	              TMOUT         §џ
                 BUSY            §џ              ERR            §џ              ERRID           §џ                       ЊK     џџџџ           ADSWRITEIND           fbAdsWriteInd        
                FW_AdsWriteInd    §џ                 CLEAR            §џ                 VALID            §џ              NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              INVOKEID           §џ	              IDXGRP           §џ
              IDXOFFS           §џ              LENGTH           §џ              DATAADDR           §џ                       ЊK      џџџџ           ADSWRITERES           fbAdsWriteRes                    FW_AdsWriteRes    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              INVOKEID           §џ              RESULT           §џ              RESPOND            §џ                           ЊK      џџџџ        	   ADSWRTCTL           fbAdsWriteControl                                FW_AdsWriteControl    §џ                 NETID            
   T_AmsNetId   §џ              PORT            	   T_AmsPort   §џ              ADSSTATE           §џ              DEVSTATE           §џ              LEN           §џ              SRCADDR           §џ              WRITE            §џ	              TMOUT         §џ
                 BUSY            §џ              ERR            §џ              ERRID           §џ                       ЊK     џџџџ           ANALYZEEXPRESSION               InputExp            §џ           	   DoAnalyze            §џ              	   ExpResult            §џ           	   OutString               §џ                       ЊK      џџџџ           ANALYZEEXPRESSIONCOMBINED           Index            §џ                 InputExp            §џ           	   DoAnalyze            §џ              	   ExpResult            §џ              OutTable   	                        ExpressionResult           §џ           	   OutString               §џ	                       ЊK      џџџџ           ANALYZEEXPRESSIONTABLE           Index            §џ                 InputExp            §џ           	   DoAnalyze            §џ              	   ExpResult            §џ              OutTable   	                        ExpressionResult           §џ                       ЊK      џџџџ           APPENDERRORSTRING               strOld               §џ              strNew               §џ                 AppendErrorString                                         ЊK      џџџџ        
   CLEARBIT32           dwConst           §џ                 inVal32           §џ              bitNo           §џ              
   CLEARBIT32                                     ЊK      џџџџ        	   CSETBIT32           dwConst           §џ                 inVal32           §џ              bitNo           §џ              bitVal            §џ       &    value to which the bit should be set    	   CSETBIT32                                     ЊK      џџџџ           DRAND           fbDRand                    FW_DRand    §џ	                 Seed           §џ                 Num                        §џ                       ЊK      џџџџ           F_COMPAREFWVERSION               major         ` §џ           requiered major version    minor         ` §џ	           requiered minor version    revision         ` §џ
       )    requiered revision/service pack version    patch         ` §џ       0    required patch version (reserved, default = 0 )      F_CompareFwVersion                                      ЊK      џџџџ           F_CREATEAMSNETID           idx         ` §џ                 nIds               T_AmsNetIdArr   §џ           Ams Net ID as array of bytes.       F_CreateAmsNetId            
   T_AmsNetId                             ЊK      џџџџ           F_CREATEIPV4ADDR           idx         ` §џ                 nIds               T_IPv4AddrArr   §џ       <    Internet Protocol dotted address (ipv4) as array of bytes.       F_CreateIPv4Addr            
   T_IPv4Addr                             ЊK      џџџџ           F_GETVERSIONTCSYSTEM               nVersionElement           §џ                 F_GetVersionTcSystem                                     ЊK      џџџџ           F_IOPORTREAD               nAddr           §џ           Port address    eSize               E_IOAccessSize   §џ           Number of bytes to read       F_IOPortRead                                     ЊK      џџџџ           F_IOPORTWRITE               nAddr           §џ           Port address    eSize               E_IOAccessSize   §џ           Number of bytes to write    nValue           §џ           Value to write       F_IOPortWrite                                      ЊK      џџџџ           F_SCANAMSNETIDS           pNetID               ` §џ              b               T_AmsNetIdArr ` §џ              w         ` §џ	              id         ` §џ
           	   Index7001                            sNetID            
   T_AmsNetID   §џ       :    String containing the Ams Net ID. E.g. '127.16.17.3.1.1'       F_ScanAmsNetIds               T_AmsNetIdArr                             ЊK      џџџџ           F_SCANIPV4ADDRIDS           b               T_AmsNetIdArr ` §џ           	   Index7001                            sIPv4            
   T_IPv4Addr   §џ       M    String containing the Internet Protocol dotted address. E.g. '172.16.7.199'       F_ScanIPv4AddrIds               T_IPv4AddrArr                             ЊK      џџџџ           F_SPLITPATHNAME           pPath                  §џ              pSlash                  §џ              pDot                  §џ              p                  §џ              length            §џ              	   sPathName               T_MaxString   §џ                 F_SplitPathName                                sDrive               §џ              sDir                T_MaxString  §џ           	   sFileName                T_MaxString  §џ              sExt                T_MaxString  §џ	                   ЊK      џџџџ           F_TOASC           pChar                  §џ                 str    Q       Q    §џ                 F_ToASC                                     ЊK      џџџџ           F_TOCHR           pChar    	                               §џ                 c           §џ                 F_ToCHR    Q       Q                              ЊK      џџџџ           FB_CREATEDIR        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id 	   sPathName               T_MaxString   §џ           max directory length = 255    ePath           PATH_GENERIC    
   E_OpenPath   §џ       +    Default: Create directory at generic path    bExecute            §џ       %    rising edge start command execution    tTimeout         §џ                 bBusy            §џ
              bError            §џ              nErrId           §џ                       ЊK     џџџџ           FB_EOF        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ              nEOF            §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ           file handle    bExecute            §џ           control input    tTimeout         §џ                 bBusy            §џ	              bError            §џ
              nErrId           §џ              bEOF            §џ                       ЊK     џџџџ           FB_FILECLOSE        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ       %    file handle obtained through 'open'    bExecute            §џ           close control input    tTimeout         §џ                 bBusy            §џ	              bError            §џ
              nErrId           §џ                       ЊK     џџџџ           FB_FILEDELETE        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ              tmpOpenMode            §џ                 sNetId            
   T_AmsNetId   §џ           ams net id 	   sPathName               T_MaxString   §џ           file path and name    ePath           PATH_GENERIC    
   E_OpenPath   §џ           Default: Open generic file    bExecute            §џ           open control input    tTimeout         §џ                 bBusy            §џ
              bError            §џ              nErrId           §џ                       ЊK     џџџџ           FB_FILEGETS        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ           file handle    bExecute            §џ           control input    tTimeout         §џ                 bBusy            §џ	              bError            §џ
              nErrId           §џ              sLine               T_MaxString   §џ              bEOF            §џ                       ЊK     џџџџ           FB_FILEOPEN        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ              tmpOpenMode            §џ              tmpHndl            §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id 	   sPathName               T_MaxString   §џ           max filename length = 255    nMode           §џ           open mode flags    ePath           PATH_GENERIC    
   E_OpenPath   §џ           Default: Open generic file    bExecute            §џ           open control input    tTimeout         §џ                 bBusy            §џ              bError            §џ              nErrId           §џ              hFile           §џ           file handle             ЊK     џџџџ           FB_FILEPUTS        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ           file handle    sLine               T_MaxString   §џ           string to write    bExecute            §џ           control input    tTimeout         §џ                 bBusy            §џ
              bError            §џ              nErrId           §џ                       ЊK     џџџџ           FB_FILEREAD        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ           file handle 	   pReadBuff           §џ           buffer address for read 	   cbReadLen           §џ           count of bytes for read    bExecute            §џ           read control input    tTimeout         §џ                 bBusy            §џ              bError            §џ              nErrId           §џ              cbRead           §џ           count of bytes actually read    bEOF            §џ                       ЊK     џџџџ           FB_FILERENAME        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ              tmpOpenMode            §џ           
   sBothNames   	                    T_MaxString            §џ           = SIZEOF( T_MaxString ) * 2    nOldLen            §џ              nNewLen            §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    sOldName               T_MaxString   §џ           max filename length = 255    sNewName               T_MaxString   §џ           max filename length = 255    ePath           PATH_GENERIC    
   E_OpenPath   §џ           Default: generic file path   bExecute            §џ           open control input    tTimeout         §џ                 bBusy            §џ              bError            §џ              nErrId           §џ                       ЊK     џџџџ           FB_FILESEEK        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ           
   tmpSeekPos   	                          §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ	           file handle    nSeekPos           §џ
           new seek pointer position    eOrigin       	    SEEK_SET       E_SeekOrigin   §џ              bExecute            §џ           seek control input    tTimeout         §џ                 bBusy            §џ              bError            §џ              nErrId           §џ                       ЊK     џџџџ           FB_FILETELL        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ           file handle    bExecute            §џ           control input    tTimeout         §џ                 bBusy            §џ	              bError            §џ
              nErrId           §џ              nSeekPos           §џ          	On error, nSEEKPOS returns -1             ЊK     џџџџ           FB_FILEWRITE        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id    hFile           §џ           file handle 
   pWriteBuff           §џ           buffer address for write 
   cbWriteLen           §џ           count of bytes for write    bExecute            §џ           write control input    tTimeout         §џ                 bBusy            §џ              bError            §џ              nErrId           §џ              cbWrite           §џ       !    count of bytes actually written             ЊK     џџџџ           FB_PCWATCHDOG           bRetVal             §џ              iTime            §џ              iIdx            §џ              iPortArr   	                 >    16#2E, 16#2E, 16#2E, 16#2F, 16#2E, 16#2F, 16#2E, 16#2F, 16#2E	      .      .      .      /      .      /      .      /      .      §џ              iArrEn   	                 >    16#87, 16#87, 16#07, 16#08, 16#F6, 16#05, 16#30, 16#01, 16#AA	                              і            0            Њ      §џ              iArrDis   	                 >    16#87, 16#87, 16#07, 16#08, 16#F6, 16#00, 16#30, 16#00, 16#AA	                              і             0             Њ      §џ                 tTimeOut           §џ       ;    Watchdog TimeOut Time 1s..255s, disabled if tTimeOut < 1s    bEnable            §џ           Enable / Disable Watchdog       bEnabled            §џ       2    TRUE: Watchdog Enabled; FALSE: Watchdog Disabled    bBusy            §џ           FB still busy    bError            §џ	           FB has error     nErrId           §џ
           FB error ID               ЊK      џџџџ           FB_REMOVEDIR        
   fbAdsRdWrt                                    FW_AdsRdWrt    §џ                 sNetId            
   T_AmsNetId   §џ           ams net id 	   sPathName               T_MaxString   §џ           max filename length = 255    ePath           PATH_GENERIC    
   E_OpenPath   §џ       +    Default: Delete directory at generic path    bExecute            §џ       &    rising edge starts command execution    tTimeout         §џ                 bBusy            §џ
              bError            §џ              nErrId           §џ                       ЊK     џџџџ           FB_SIMPLEADSLOGEVENT           fbEvent                            ADSLOGEVENT    §џ              cfgEvent               TcEvent    §џ              bInit            §џ                 SourceID           §џ              EventID           §џ           	   bSetEvent           §џ              bQuit            §џ                 ErrId           §џ	              Error            §џ
                       ЊK      џџџџ        	   FILECLOSE        
   fbAdsWrite                                FW_AdsWrite    §џ                 NETID            
   T_AmsNetId   §џ           ams net id    HFILE           §џ       )    file handle obtained through 'FILEOPEN'    CLOSE            §џ           close control input    TMOUT         §џ                 BUSY            §џ	              ERR            §џ
              ERRID           §џ                       ЊK     џџџџ           FILEOPEN        
   fbAdsWrite                                FW_AdsWrite    §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ                 NETID            
   T_AmsNetId   §џ           ams net id 	   FPATHNAME               T_MaxString   §џ       #    default max filename length = 255    OPENMODE           §џ           open mode flags    OPEN            §џ           open control input    TMOUT         §џ                 BUSY            §џ
              ERR            §џ              ERRID           §џ              HFILE           §џ           file handle             ЊK     џџџџ           FILEREAD        	   fbAdsRead                              
   FW_AdsRead    §џ                 NETID            
   T_AmsNetId   §џ           ams net id    HFILE           §џ           file handle    BUFADDR           §џ           buffer address for read    COUNT           §џ           count of bytes for read    READ            §џ           read control input    TMOUT         §џ                 BUSY            §џ              ERR            §џ              ERRID           §џ              COUNT_R           §џ           count of bytes actually read             ЊK     џџџџ           FILESEEK        
   fbAdsWrite                                FW_AdsWrite    §џ                 NETID            
   T_AmsNetId   §џ           ams net id    HFILE           §џ           file handle    SEEKPOS           §џ           new seek pointer position    SEEK            §џ           seek control input    TMOUT         §џ                 BUSY            §џ
              ERR            §џ              ERRID           §џ                       ЊK     џџџџ        	   FILEWRITE        
   fbAdsWrite                                FW_AdsWrite    §џ           
   RisingEdge                 R_TRIG    §џ              FallingEdge                 F_TRIG    §џ              tmpCount            §џ                 NETID            
   T_AmsNetId   §џ           ams net id    HFILE           §џ           file handle    BUFADDR           §џ           buffer address for write    COUNT           §џ           count of bytes for write    WRITE            §џ           write control input    TMOUT         §џ                 BUSY            §џ              ERR            §џ              ERRID           §џ              COUNT_W           §џ       !    count of bytes actually written             ЊK     џџџџ           FW_CALLGENERICFB           fbCall       w    ( 	sNetID := '', nPort := 16#1234,
								bExecute := FALSE, tTimeout := T#0s,
								ACCESSCNT_I := 16#0000BEC1 )     СО                 4                          FW_AdsRdWrt ` §џ                 funGrp         ` §џ       #    Function block group (identifier)    funNum         ` §џ       $    Function block number (identifier)    pWrite         ` §џ       +    Byte length of output parameter structure    cbWrite         ` §џ       *    Byte length of input parameter structure    pRead         ` §џ	           Points ot output buffer    cbRead         ` §џ
           Points to input buffer       nErrID         ` §џ           0 => no error, <> 0 => error
   cbReturned         ` §џ       ,    Number of successfully returned data bytes             ЊK      џџџџ           FW_CALLGENERICFUN           fbCall       y    ( 	sNetID := '', nPort := 16#1234,
									bExecute := FALSE, tTimeout := T#0s,
									ACCESSCNT_I := 16#0000BEC2 )     ТО                 4                          FW_AdsRdWrt ` §џ           don't use it!        funGrp         ` §џ           Function group (identifier)    funNum         ` §џ       $    Function block number (identifier)    pWrite         ` §џ       +    Byte length of output parameter structure    cbWrite         ` §џ	       *    Byte length of input parameter structure    pRead         ` §џ
           Points ot output buffer    cbRead         ` §џ           Points to input buffer    pcbReturned               ` §џ       ,    Number of successfully returned data bytes       FW_CallGenericFun                                     ЊK      џџџџ           GETBIT32           dwConst           §џ                 inVal32           §џ              bitNo           §џ                 GETBIT32                                      ЊK      џџџџ           GETCPUACCOUNT           fbGetCpuAccount               FW_GetCpuAccount    §џ                     cpuAccountDW           §џ                       ЊK      џџџџ           GETCPUCOUNTER           fbGetCpuCounter                FW_GetCpuCounter    §џ	                  
   cpuCntLoDW           §џ           
   cpuCntHiDW           §џ                       ЊK      џџџџ           GETCURTASKINDEX           fbGetCurTaskIndex               FW_GetCurTaskIndex    §џ                     index           §џ                       ЊK      џџџџ           GETSYSTEMTIME           fbGetSystemTime                FW_GetSystemTime    §џ	                     timeLoDW           §џ              timeHiDW           §џ                       ЊK      џџџџ           GETTASKTIME           out   	                       ` §џ	           
   cbReturned         ` §џ
                     timeLoDW           §џ              timeHiDW           §џ                       ЊK      џџџџ        	   LPTSIGNAL               PortAddr           §џ              PinNo           §џ              OnOff            §џ	              	   LPTSIGNAL                                      ЊK      џџџџ           MEMCMP               pBuf1           §џ           First buffer    pBuf2           §џ           Second buffer    n           §џ           Number of characters       MEMCMP                                     ЊK      џџџџ           MEMCPY               destAddr           §џ           New buffer    srcAddr           §џ           Buffer to copy from    n           §џ           Number of characters to copy       MEMCPY                                     ЊK      џџџџ           MEMMOVE               destAddr           §џ           New buffer    srcAddr           §џ           Buffer to copy from    n           §џ           Number of characters to copy       MEMMOVE                                     ЊK      џџџџ           MEMSET               destAddr           §џ           Pointer to destination    fillByte           §џ           Character to set    n           §џ           Number of characters       MEMSET                                     ЊK      џџџџ           ROL32               inVal32           §џ              n           §џ                 ROL32                                     ЊK      џџџџ           ROR32               inVal32           §џ              n           §џ                 ROR32                                     ЊK      џџџџ           SETBIT32           dwConst           §џ                 inVal32           §џ              bitNo           §џ                 SETBIT32                                     ЊK      џџџџ           SFCACTIONCONTROL     
      S_FF                 RS    §џ              L_TMR                    TON    §џ              D_TMR                    TON    §џ              P_TRIG                 R_TRIG    §џ              SD_TMR                    TON    §џ              SD_FF                 RS    §џ              DS_FF                 RS    §џ              DS_TMR                    TON    §џ              SL_FF                 RS    §џ              SL_TMR                    TON    §џ           
      N            §џ              R0            §џ              S0            §џ              L            §џ              D            §џ              P            §џ              SD            §џ	              DS            §џ
              SL            §џ              T           §џ                 Q            §џ                       ЊK      џџџџ           SHL32               inVal32           §џ              n           §џ                 SHL32                                     ЊK      џџџџ           SHR32               inVal32           §џ              n           §џ                 SHR32                                     ЊK      џџџџ    o   C:\TWINCAT\PLC\LIB\TcBase.lib @                                                                                          FW_ADSCLEAREVENTS           STAMP_I            §џ              ACCESSCNT_I            §џ              BUSY_I             §џ              ERR_I             §џ              ERRID_I            §џ           
   READ_SAV_I             §џ              TICKSTART_I            §џ                 sNetId               §џ              bClear            §џ              nMode           §џ              tTimeout           §џ                 bBusy            §џ	              bError            §џ
              nErrId           §џ                       `QJ     џџџџ           FW_ADSLOGDINT            	   nCtrlMask           §џ              sMsgFmt               §џ              nArg           §џ                 FW_AdsLogDINT                                     `QJ     џџџџ           FW_ADSLOGEVENT        
   STAMPREQ_I            §џ           
   STAMPRES_I            §џ           
   STAMPSIG_I            §џ           
   STAMPCON_I            §џ              ACCESSCNT_I            §џ           	   AMSADDR_I   	                         §џ              EVENT_I                      
   FW_TcEvent    §џ              pTCEVENTSTREAM_I            §џ              CBEVENTSTREAM_I            §џ              nSTATE_I            §џ              nSTATEREQ_I            §џ              nSTATERES_I            §џ              nSTATESIG_I            §џ               nSTATECON_I            §џ!              ERR_I             §џ"              ERRID_I            §џ#              bEVENT_SAV_I             §џ$              bEVENTQUIT_SAV_I             §џ%              TICKSTART_I            §џ&           	      sNetId               §џ              nPort           §џ              bEvent            §џ           
   bEventQuit            §џ              stEventConfigData                      
   FW_TcEvent   §џ              pEventDataAddress           §џ       	    pointer    cbEventDataLength           §џ	           
   bFbCleanup            §џ
              tTimeout           §џ                 nEventState           §џ              bError            §џ              nErrId           §џ              bQuit            §џ                       `QJ     џџџџ           FW_ADSLOGLREAL            	   nCtrlMask           §џ              sMsgFmt               §џ              fArg                        §џ                 FW_AdsLogLREAL                                     `QJ     џџџџ           FW_ADSLOGSTR            	   nCtrlMask           §џ              sMsgFmt               §џ              sArg               §џ                 FW_AdsLogSTR                                     `QJ     џџџџ           FW_ADSRDWRT           STAMP_I            §џ              ACCESSCNT_I            §џ              BUSY_I             §џ              ERR_I             §џ              ERRID_I            §џ              WRTRD_SAV_I             §џ              PDESTADDR_I            §џ              TICKSTART_I            §џ           
      sNetId               §џ              nPort           §џ              nIdxGrp           §џ              nIdxOffs           §џ           
   cbWriteLen           §џ           	   cbReadLen           §џ           
   pWriteBuff           §џ	           	   pReadBuff           §џ
              bExecute            §џ              tTimeout           §џ                 bBusy            §џ              bError            §џ              nErrId           §џ              cbRead           §џ           count of bytes actually read             `QJ     џџџџ           FW_ADSRDWRTIND           CLEAR_I             §џ                 bClear            §џ           	      bValid            §џ              sNetId               §џ              nPort           §џ           	   nInvokeId           §џ	              nIdxGrp           §џ
              nIdxOffs           §џ           	   cbReadLen           §џ           
   cbWriteLen           §џ           
   pWriteBuff           §џ                       `QJ     џџџџ           FW_ADSRDWRTRES        	   RESPOND_I             §џ                 sNetId               §џ              nPort           §џ           	   nInvokeId           §џ              nErrId           §џ           	   cbReadLen           §џ           	   pReadBuff           §џ              bRespond            §џ	                           `QJ     џџџџ        
   FW_ADSREAD           STAMP_I            §џ              ACCESSCNT_I            §џ              BUSY_I             §џ              ERR_I             §џ              ERRID_I            §џ           
   READ_SAV_I             §џ              TICKSTART_I            §џ                 sNetId               §џ              nPort           §џ              nIdxGrp           §џ              nIdxOffs           §џ           	   cbReadLen           §џ           	   pReadBuff           §џ              bExecute            §џ	              tTimeout           §џ
                 bBusy            §џ              bError            §џ              nErrId           §џ              cbRead           §џ           count of bytes actually read             `QJ     џџџџ           FW_ADSREADDEVICEINFO           STAMP_I            §џ              ACCESSCNT_I            §џ              BUSY_I             §џ              ERR_I             §џ              ERRID_I            §џ              RDINFO_SAV_I             §џ              TICKSTART_I            §џ                 sNetId               §џ              nPort           §џ              bExecute            §џ              tTimeout           §џ                 bBusy            §џ	              bError            §џ
              nErrId           §џ              sDevName               §џ              nDevVersion           §џ                       `QJ     џџџџ           FW_ADSREADIND           CLEAR_I             §џ                 bClear            §џ                 bValid            §џ              sNetId               §џ              nPort           §џ           	   nInvokeId           §џ	              nIdxGrp           §џ
              nIdxOffs           §џ           	   cbReadLen           §џ                       `QJ     џџџџ           FW_ADSREADRES        	   RESPOND_I             §џ                 sNetId               §џ              nPort           §џ           	   nInvokeId           §џ              nErrId           §џ           	   cbReadLen           §џ           	   pReadBuff           §џ              bRespond            §џ	                           `QJ     џџџџ           FW_ADSREADSTATE           STAMP_I            §џ              ACCESSCNT_I            §џ              BUSY_I             §џ              ERR_I             §џ              ERRID_I            §џ              RDSTATE_SAV_I             §џ              TICKSTART_I            §џ                 sNetId               §џ              nPort           §џ              bExecute            §џ              tTimeout           §џ                 bBusy            §џ	              bError            §џ
              nErrId           §џ           	   nAdsState           §џ           	   nDevState           §џ                       `QJ     џџџџ           FW_ADSWRITE           STAMP_I            §џ              ACCESSCNT_I            §џ              BUSY_I             §џ              ERR_I             §џ              ERRID_I            §џ              WRITE_SAV_I             §џ              TICKSTART_I            §џ                 sNetId               §џ              nPort           §џ              nIdxGrp           §џ              nIdxOffs           §џ           
   cbWriteLen           §џ           
   pWriteBuff           §џ              bExecute            §џ	              tTimeout           §џ
                 bBusy            §џ              bError            §џ              nErrId           §џ                       `QJ     џџџџ           FW_ADSWRITECONTROL           STAMP_I            §џ              ACCESSCNT_I            §џ              BUSY_I             §џ              ERR_I             §џ              ERRID_I            §џ              WRITE_SAV_I             §џ              TICKSTART_I            §џ                 sNetId               §џ              nPort           §џ           	   nAdsState           §џ           	   nDevState           §џ           
   cbWriteLen           §џ           
   pWriteBuff           §џ              bExecute            §џ	              tTimeout           §џ
                 bBusy            §џ              bError            §џ              nErrId           §џ                       `QJ     џџџџ           FW_ADSWRITEIND           CLEAR_I             §џ                 bClear            §џ                 bValid            §џ              sNetId               §џ              nPort           §џ           	   nInvokeId           §џ	              nIdxGrp           §џ
              nIdxOffs           §џ           
   cbWriteLen           §џ           
   pWriteBuff           §џ                       `QJ     џџџџ           FW_ADSWRITERES        	   RESPOND_I             §џ                 sNetId               §џ              nPort           §џ           	   nInvokeId           §џ              nErrId           §џ              bRespond            §џ                           `QJ     џџџџ           FW_DRAND           FirstCall_i             §џ	           
   HoldRand_i            §џ
              R250_Buffer_i   	  љ                        §џ           
   R250_Index            §џ                 nSeed           §џ                 fRndNum                        §џ                       `QJ     џџџџ           FW_GETCPUACCOUNT                   dwCpuAccount           §џ                       `QJ     џџџџ           FW_GETCPUCOUNTER                
   dwCpuCntLo           §џ           
   dwCpuCntHi           §џ                       `QJ     џџџџ           FW_GETCURTASKINDEX                   nIndex           §џ                       `QJ     џџџџ           FW_GETSYSTEMTIME                   dwTimeLo           §џ              dwTimeHi           §џ                       `QJ     џџџџ           FW_GETVERSIONTCBASE               nVersionElement           §џ                 FW_GetVersionTcBase                                     `QJ     џџџџ           FW_LPTSIGNAL            	   nPortAddr           §џ              nPinNo           §џ              bOnOff            §џ	                 FW_LptSignal                                      `QJ     џџџџ        	   FW_MEMCMP               pBuf1           §џ           First buffer    pBuf2           §џ           Second buffer    cbLen           §џ           Number of characters    	   FW_MemCmp                                     `QJ     џџџџ        	   FW_MEMCPY               pDest           §џ           New buffer    pSrc           §џ           Buffer to copy from    cbLen           §џ           Number of characters to copy    	   FW_MemCpy                                     `QJ     џџџџ        
   FW_MEMMOVE               pDest           §џ           New buffer    pSrc           §џ           Buffer to copy from    cbLen           §џ           Number of characters to copy    
   FW_MemMove                                     `QJ     џџџџ        	   FW_MEMSET               pDest           §џ           Pointer to destination 	   nFillByte           §џ           Character to set    cbLen           §џ           Number of characters    	   FW_MemSet                                     `QJ     џџџџ           FW_PORTREAD            	   nPortAddr           §џ           	   eNoOfByte               FW_NoOfByte   §џ                 FW_PortRead                                     `QJ     џџџџ           FW_PORTWRITE            	   nPortAddr           §џ           	   eNoOfByte               FW_NoOfByte   §џ              nValue           §џ                 FW_PortWrite                                      `QJ     џџџџ    q   C:\TWINCAT\PLC\LIB\STANDARD.LIB @                                                                                          CONCAT               STR1               §џ              STR2               §џ                 CONCAT                                         DN76     џџџџ           CTD           M             §џ           Variable for CD Edge Detection      CD            §џ           Count Down on rising edge    LOAD            §џ           Load Start Value    PV           §џ           Start Value       Q            §џ           Counter reached 0    CV           §џ           Current Counter Value             DN76     џџџџ           CTU           M             §џ            Variable for CU Edge Detection       CU            §џ       
    Count Up    RESET            §џ           Reset Counter to 0    PV           §џ           Counter Limit       Q            §џ           Counter reached the Limit    CV           §џ           Current Counter Value             DN76     џџџџ           CTUD           MU             §џ            Variable for CU Edge Detection    MD             §џ            Variable for CD Edge Detection       CU            §џ	       
    Count Up    CD            §џ
           Count Down    RESET            §џ           Reset Counter to Null    LOAD            §џ           Load Start Value    PV           §џ           Start Value / Counter Limit       QU            §џ           Counter reached Limit    QD            §џ           Counter reached Null    CV           §џ           Current Counter Value             DN76     џџџџ           DELETE               STR               §џ              LEN           §џ              POS           §џ                 DELETE                                         DN76     џџџџ           F_TRIG           M             §џ
                 CLK            §џ           Signal to detect       Q            §џ           Edge detected             DN76     џџџџ           FIND               STR1               §џ              STR2               §џ                 FIND                                     DN76     џџџџ           INSERT               STR1               §џ              STR2               §џ              POS           §џ                 INSERT                                         DN76     џџџџ           LEFT               STR               §џ              SIZE           §џ                 LEFT                                         DN76     џџџџ           LEN               STR               §џ                 LEN                                     DN76     џџџџ           MID               STR               §џ              LEN           §џ              POS           §џ                 MID                                         DN76     џџџџ           R_TRIG           M             §џ
                 CLK            §џ           Signal to detect       Q            §џ           Edge detected             DN76     џџџџ           REPLACE               STR1               §џ              STR2               §џ              L           §џ              P           §џ                 REPLACE                                         DN76     џџџџ           RIGHT               STR               §џ              SIZE           §џ                 RIGHT                                         DN76     џџџџ           RS               SET            §џ              RESET1            §џ                 Q1            §џ
                       DN76     џџџџ           SEMA           X             §џ                 CLAIM            §џ	              RELEASE            §џ
                 BUSY            §џ                       DN76     џџџџ           SR               SET1            §џ              RESET            §џ                 Q1            §џ	                       DN76     џџџџ           TOF           M             §џ           internal variable 	   StartTime            §џ           internal variable       IN            §џ       ?    starts timer with falling edge, resets timer with rising edge    PT           §џ           time to pass, before Q is set       Q            §џ	       2    is FALSE, PT seconds after IN had a falling edge    ET           §џ
           elapsed time             DN76     џџџџ           TON           M             §џ           internal variable 	   StartTime            §џ           internal variable       IN            §џ       ?    starts timer with rising edge, resets timer with falling edge    PT           §џ           time to pass, before Q is set       Q            §џ	       0    is TRUE, PT seconds after IN had a rising edge    ET           §џ
           elapsed time             DN76     џџџџ           TP        	   StartTime            §џ           internal variable       IN            §џ       !    Trigger for Start of the Signal    PT           §џ       '    The length of the High-Signal in 10ms       Q            §џ	           The pulse    ET           §џ
       &    The current phase of the High-Signal             DN76     џџџџ    R    @                                                                                          FB_FIFO           MaxFifoSize       @   
              FifoEntries   	                      ST_FifoEntry                           nFirst                          nLast                          nCount                              new                 ST_FifoEntry                     bOk                           old                 ST_FifoEntry                           мIL  @   џџџџ           FB_PEERTOPEER           fbCreate                            FB_SocketUdpCreate    &               fbClose        
                FB_SocketClose    &               fbReceiveFrom                              FB_SocketUdpReceiveFrom    &               fbSendTo                             FB_SocketUdpSendTo    &               hSocket              	   T_HSOCKET    &               eStep               E_ClientServerSteps    &               sendTo                 ST_FifoEntry    &               receivedFrom                 ST_FifoEntry    &               
   sLocalHost               &            
   nLocalPort           & 	              bEnable            & 
                 bCreated            &               bBusy            &               bError            &               nErrId           &                  sendFifo                       FB_Fifo  &               receiveFifo                       FB_Fifo  &                    мIL  @    џџџџ           LOGERROR               msg    Q       Q    #               nErrId           #                  LogError                                     мIL  @    џџџџ        
   LOGMESSAGE               msg    Q       Q    $               hSocket              	   T_HSOCKET   $               
   LogMessage                                     мIL  @    џџџџ           MAIN           LOCAL_HOST_IP                               LOCAL_HOST_PORT    ъ                     REMOTE_HOST_IP          192.168.1.108                    REMOTE_HOST_PORT    щ                     T1                    TON      
           
   sendPacket                             I                            Inputs   	                                         Outputs   	                                          TimeVar    !                    networking variables    fbSocketCloseAll        	               FB_SocketCloseAll                 	   bCloseAll                            fbPeerToPeer                               FB_PeerToPeer                    sendFifo                      FB_Fifo                    receiveFifo                      FB_Fifo                    sendToEntry                 ST_FifoEntry                    entryReceivedFrom                 ST_FifoEntry                    tmp    Q       Q                     bSendOnceToItself                             bSendOnceToRemote                                              p`L  @  џџџџ        
   SCODE_CODE               sc           %               
   SCODE_CODE                                     мIL  @    џџџџ            
 ў       (      K       K   )    K   7    K   L                Y        +     КЛlocalhost       уТТw   Јѓ@            Ј    ай     pл \Уwp СwџџџџуТТw>2     шћ Јѓ@        @№џџЈѓ@     рд к Ад\        Ад   Идb@Ј   џџ    \и Фк Јк ю|№|џџџџы|шћ Јѓ@        шћ Јѓ@     фу `Oџџџџpл фу xOџџџџ|л Н8у     ,   ,                                                        K     )   F:\UPD comm in Twincat\PeerToPeerA.pro @мILT /*BECKCONFI3*/
        !ЮЛ @   @           3               
   Standard            	p`L                        VAR_GLOBAL
END_VAR
                                                                                  "   , X h §e             Standard
         MAINџџџџ               p`L                 $ћџџџ                                            Standard М>L	М>L                                       	p`L                        VAR_CONFIG
END_VAR
                                                                                   '              , јџ }           Global_Variables мIL	p`L     Дrџџ             VAR_GLOBAL
	g_sTcIpConnSvrAddr								:  T_AmsNetId := '';
	bLogDebugMessages							: BOOL := TRUE;

	(* Some project specific error codes *)
	PLCPRJ_ERROR_SENDFIFO_OVERFLOW		: UDINT := 16#8103;
	PLCPRJ_ERROR_RECFIFO_OVERFLOW		: UDINT := 16#8104;
END_VAR                                                                                               '           Н     epes ske           TwinCAT_Configuration мIL	p`LН     Дrџџ           Ј   (* Generated automatically by TwinCAT - (read only) *)
VAR_CONFIG
	MAIN.Inputs AT %IB0 : ARRAY [1..2] OF BOOL;
	MAIN.Outputs AT %QB0 : ARRAY [1..2] OF BOOL;
END_VAR                                                                                               '           	   , , } бz           Variable_Configuration мIL	мIL	     Дrџџ              VAR_CONFIG
END_VAR
                                                                                                    |0|0 @v    @T   MS Sans Serif @       HH':'mm':'ss   dd'-'MM'-'yyyy   dd'-'MM'-'yyyy HH':'mm':'ssѓџџџ                               њ      џ   џџџ  Ь3 џџџ   џ џџџ                  DEFAULT             System         |0|0 @v    @T   MS Sans Serif @       HH':'mm':'ss   dd'-'MM'-'yyyy   dd'-'MM'-'yyyy HH':'mm':'ssѓџџџ                         HH':'mm':'ss   dd'-'MM'-'yyyy'          П   , Ц ъ kч           E_ClientServerSteps мIL	мIL      6#03
	CP          TYPE E_ClientServerSteps :
(
	UDP_STATE_IDLE		:= 0,
	UDP_STATE_CREATE_START,
	UDP_STATE_CREATE_WAIT,
	UDP_STATE_SEND_START,
	UDP_STATE_SEND_WAIT,
	UDP_STATE_RECEIVE_START,
	UDP_STATE_RECEIVE_WAIT,
	UDP_STATE_CLOSE_START,
	UDP_STATE_CLOSE_WAIT,
	UDP_STATE_ERROR
);
END_TYPE             Р   ,     R§           ST_FifoEntry мIL	мIL       3CwAC        6  TYPE ST_FifoEntry :
STRUCT
	sRemoteHost		: STRING(15);		(* Remote address. String containing an (Ipv4) Internet Protocol dotted address. *)
	nRemotePort		: UDINT;			(* Remote Internet Protocol (IP) port. *)
	msg				: STRING;			(* Udp packet data *)
	(*msg				: ARRAY[1..2] OF BOOL;*)
END_STRUCT
END_TYPE                 , Z ц џ!           FB_Fifo мIL	мIL      unr  P
        P  FUNCTION_BLOCK FB_Fifo
VAR_INPUT
	new					: ST_FifoEntry;
END_VAR
VAR_OUTPUT
	bOk					: BOOL;
	old						: ST_FifoEntry;
END_VAR
VAR CONSTANT
	MaxFifoSize			: INT := 5;
END_VAR
VAR
	FifoEntries				: ARRAY[1..MaxFifoSize] OF ST_FifoEntry;
	nFirst					: BYTE := 1;
	nLast 					: BYTE := 1;
	nCount				: BYTE := 0;
END_VAR
   ;    , h х т           AddTail мILК   IF nCount >= MaxFifoSize THEN
	bOk := FALSE;
	RETURN;
END_IF

FifoEntries[ nLast ] := new;
nLast := SEL( nLast = MaxFifoSize, nLast + 1, 1 );

nCount := nCount + 1;
bOk := TRUE;"   , Y И ўЕ        
   RemoveHead мILБ   IF nCount = 0 THEN
	bOk := FALSE;
	RETURN;
END_IF

old := FifoEntries[ nFirst ];
nFirst := SEL( nFirst = MaxFifoSize, nFirst + 1, 1 );
nCount := nCount - 1;
bOk := TRUE;             &   , d H 	Х           FB_PeerToPeer мIL	мIL          e            FUNCTION_BLOCK FB_PeerToPeer
(* Function block example of udp peer-to-peer application *)
VAR_IN_OUT
	sendFifo				: FB_Fifo;
	receiveFifo				: FB_Fifo;
END_VAR
VAR_INPUT
	sLocalHost			: STRING(15);
	nLocalPort				: UDINT;
	bEnable 				: BOOL;
END_VAR
VAR_OUTPUT
	bCreated				: BOOL;
	bBusy 					: BOOL;
	bError 					: BOOL;
	nErrId 					: UDINT;
END_VAR
VAR
	fbCreate				: FB_SocketUdpCreate;
	fbClose				: FB_SocketClose;
	fbReceiveFrom		: FB_SocketUdpReceiveFrom;
	fbSendTo				: FB_SocketUdpSendTo;
	hSocket 				: T_HSOCKET;
	eStep					: E_ClientServerSteps;
	sendTo				: ST_FifoEntry;
	receivedFrom			: ST_FifoEntry;
END_VARh  CASE eStep OF
	UDP_STATE_IDLE:
		IF bEnable XOR bCreated THEN
			bBusy 					:= TRUE;
			bError 					:= FALSE;
			nErrid 					:= 0;
			IF bEnable THEN
				eStep := UDP_STATE_CREATE_START;
			ELSE
				eStep := UDP_STATE_CLOSE_START;
			END_IF
		ELSIF bCreated THEN
			sendFifo.RemoveHead( old => sendTo );
			IF sendFifo.bOk THEN
				eStep := UDP_STATE_SEND_START;
			ELSE (* empty *)
				eStep := UDP_STATE_RECEIVE_START;
			END_IF
		ELSE
			bBusy := FALSE;
		END_IF

	UDP_STATE_CREATE_START:
		fbCreate(  bExecute := FALSE  );
		fbCreate(	sSrvNetId:= g_sTcIpConnSvrAddr,
					sLocalHost:= sLocalHost,
					nLocalPort:= nLocalPort,
					bExecute:= TRUE );
		eStep := UDP_STATE_CREATE_WAIT;

	UDP_STATE_CREATE_WAIT:
		fbCreate( bExecute := FALSE );
		IF NOT fbCreate.bBusy THEN
			IF NOT fbCreate.bError THEN
				bCreated := TRUE;
				hSocket := fbCreate.hSocket;
				eStep := UDP_STATE_IDLE;
				LogMessage( 'Socket opened (UDP)!', hSocket );
			ELSE
				LogError( 'FB_SocketUdpCreate', fbCreate.nErrId );
				nErrId := fbCreate.nErrId;
				eStep := UDP_STATE_ERROR;
			END_IF
		END_IF

	UDP_STATE_SEND_START:
		fbSendTo( bExecute := FALSE );
		fbSendTo(	sSrvNetId:=g_sTcIpConnSvrAddr,
					sRemoteHost := sendTo.sRemoteHost,
					nRemotePort := sendTo.nRemotePort,
					hSocket:= hSocket,
					pSrc:= ADR( sendTo.msg ),
					cbLen:= LEN( sendTo.msg ) + 1, (* include the end delimiter *)
					bExecute:= TRUE );
		eStep := UDP_STATE_SEND_WAIT;

	UDP_STATE_SEND_WAIT:
		fbSendTo( bExecute := FALSE );
		IF NOT fbSendTo.bBusy THEN
			IF NOT fbSendTo.bError THEN
				eStep := UDP_STATE_RECEIVE_START;
			ELSE
				LogError( 'FB_SocketSendTo (UDP)', fbSendTo.nErrId );
				nErrId := fbSendTo.nErrId;
				eStep := UDP_STATE_ERROR;
			END_IF
		END_IF

	UDP_STATE_RECEIVE_START:
		MEMSET( ADR( receivedFrom ), 0, SIZEOF( receivedFrom ) );
		fbReceiveFrom( bExecute := FALSE );
		fbReceiveFrom(	sSrvNetId:=g_sTcIpConnSvrAddr,
							hSocket:= hSocket,
							pDest:= ADR( receivedFrom.msg ),
							cbLen:= SIZEOF( receivedFrom.msg ) - 1, (*without string delimiter *)
							bExecute:= TRUE );
		eStep := UDP_STATE_RECEIVE_WAIT;

	UDP_STATE_RECEIVE_WAIT:
		fbReceiveFrom( bExecute := FALSE );
		IF NOT fbReceiveFrom.bBusy THEN
			IF NOT fbReceiveFrom.bError THEN
				IF fbReceiveFrom.nRecBytes > 0 THEN
					receivedFrom.nRemotePort := fbReceiveFrom.nRemotePort;
					receivedFrom.sRemoteHost := fbReceiveFrom.sRemoteHost;
					receiveFifo.AddTail( new := receivedFrom );
					IF NOT receiveFifo.bOk THEN(* Check for fifo overflow *)
						LogError( 'Receive fifo overflow!', PLCPRJ_ERROR_RECFIFO_OVERFLOW );
					END_IF
				END_IF
				eStep := UDP_STATE_IDLE;
			ELSIF fbReceiveFrom.nErrId = 16#80072746 THEN
				LogError( 'The connection is reset by remote side.', fbReceiveFrom.nErrId );
				eStep := UDP_STATE_IDLE;
			ELSE
				LogError( 'FB_SocketUdpReceiveFrom (UDP client/server)', fbReceiveFrom.nErrId );
				nErrId := fbReceiveFrom.nErrId;
				eStep := UDP_STATE_ERROR;
			END_IF
		END_IF

	UDP_STATE_CLOSE_START:
		fbClose( bExecute := FALSE );
		fbClose(	sSrvNetId:= g_sTcIpConnSvrAddr,
					hSocket:= hSocket,
					bExecute:= TRUE );
		eStep := UDP_STATE_CLOSE_WAIT;

	UDP_STATE_CLOSE_WAIT:
		fbClose( bExecute := FALSE );
		IF NOT fbClose.bBusy THEN
			LogMessage( 'Socket closed (UDP)!', hSocket );
			bCreated := FALSE;
			MEMSET( ADR(hSocket), 0, SIZEOF(hSocket));
			IF fbClose.bError THEN
				LogError( 'FB_SocketClose (UDP)', fbClose.nErrId );
				nErrId := fbClose.nErrId;
				eStep := UDP_STATE_ERROR;
			ELSE
				bBusy := FALSE;
				bError := FALSE;
				nErrId := 0;
				eStep := UDP_STATE_IDLE;
			END_IF
		END_IF

	UDP_STATE_ERROR: (* Error step *)
		bError := TRUE;
		IF bCreated THEN
			eStep := UDP_STATE_CLOSE_START;
		ELSE
			bBusy := FALSE;
			eStep := UDP_STATE_IDLE;
		END_IF
END_CASE
               #   , А а UЭ           LogError мIL	мIL                  `   FUNCTION LogError : DINT
VAR_INPUT
	msg			: STRING;
	nErrId			: DWORD;
END_VAR
VAR
END_VAR  IF bLogDebugMessages THEN
	IF nErrId = 0 THEN
		LogError := ADSLOGSTR( ADSLOG_MSGTYPE_HINT OR ADSLOG_MSGTYPE_LOG, CONCAT( msg, '       No error!   %s'),'' );
	ELSIF ( nErrId AND 16#80000000) =16#80000000 THEN
		LogError := ADSLOGDINT( ADSLOG_MSGTYPE_ERROR OR ADSLOG_MSGTYPE_LOG, CONCAT( msg, '       Win32 error: %d' ), SCODE_CODE( nErrId ) );
	ELSIF (nErrId AND 16#00008100) =16#00008100 THEN
		LogError := ADSLOGDINT( ADSLOG_MSGTYPE_ERROR OR ADSLOG_MSGTYPE_LOG, CONCAT( msg, '       Internal PLC sample project error: %d' ), nErrId );
	ELSIF (nErrId AND 16#00008000) =16#00008000 THEN
		LogError := ADSLOGDINT( ADSLOG_MSGTYPE_ERROR OR ADSLOG_MSGTYPE_LOG, CONCAT( msg, '       Internal TCP/IP Connection Server error: %d' ), nErrId );
	ELSE
		LogError := ADSLOGDINT( ADSLOG_MSGTYPE_ERROR OR ADSLOG_MSGTYPE_LOG, CONCAT( msg, '       TwinCAT System error: %d' ), nErrId );
	END_IF
END_IF               $   , Ц ъ kч        
   LogMessage мIL	мIL      pЫ §џ         d   FUNCTION LogMessage : DINT
VAR_INPUT
	msg		: STRING;
	hSocket	: T_HSOCKET;
END_VAR
VAR
END_VARЉ   IF bLogDebugMessages THEN
	LogMessage := ADSLOGDINT( ADSLOG_MSGTYPE_HINT OR ADSLOG_MSGTYPE_LOG, CONCAT( msg, '        Internal handle: %d' ), hSocket.handle  );
END_IF                   ,   Го           MAIN p`L	p`L        !№         
  PROGRAM MAIN
VAR CONSTANT
	LOCAL_HOST_IP				: STRING(15) 		:= '';
	LOCAL_HOST_PORT			: UDINT			:= 1002;
	REMOTE_HOST_IP			: STRING(15) 		:= '192.168.1.108';
	REMOTE_HOST_PORT		: UDINT 			:= 1001;
END_VAR
VAR
	(* timer and hardware variables *)
	T1: TON;
	sendPacket: BOOL;
	I: INT;
	Inputs AT %I* : ARRAY [1..2] OF BOOL;
	Outputs AT %Q* : ARRAY [1..2] OF BOOL;
	TimeVar: TIME := t#33ms;

	(* networking variables *)
	fbSocketCloseAll 				: FB_SocketCloseAll;
	bCloseAll: BOOL := TRUE;
	fbPeerToPeer				: FB_PeerToPeer;
    	sendFifo						: FB_Fifo;
    	receiveFifo					: FB_Fifo;
    	sendToEntry					: ST_FifoEntry;
    	entryReceivedFrom			: ST_FifoEntry;
    	tmp							: STRING;

	bSendOnceToItself			: BOOL;
	bSendOnceToRemote			: BOOL;
END_VAR)  IF bCloseAll THEN (*On PLC reset or program download close all old connections *)
	bCloseAll := FALSE;
	fbSocketCloseAll( sSrvNetId:= g_sTcIpConnSvrAddr, bExecute:= TRUE, tTimeout:= T#10s );
ELSE
	fbSocketCloseAll( bExecute:= FALSE );
END_IF

(* timer set for reading sensor values and sending out a packet *)
T1
(IN:=TRUE AND NOT t1.Q,
 PT:=TimeVar,
 Q=>sendPacket ,
 ET=> );

IF sendPacket THEN
	FOR I:=1 TO 2 BY 1 DO
		(*Outputs[I] := TRUE;*)
		Outputs[I] := Inputs[I];
	END_FOR
END_IF


IF NOT fbSocketCloseAll.bBusy AND NOT fbSocketCloseAll.bError THEN

	IF sendPacket THEN
		(* read in sensor values and send out UDP packet *)
		sendToEntry.nRemotePort 		:= 1001;			(* remote host port number*)
		sendToEntry.sRemoteHost 		:= '192.168.1.108';				(* remote host IP address *)
		sendToEntry.msg 				:= '';							(* start blank text*);
		FOR I:=1 TO 2 BY 1 DO
			IF Inputs[I] THEN
				sendToEntry.msg 			:=  CONCAT(sendToEntry.msg, '1');					(* message text*);
			ELSE
				sendToEntry.msg 			:=  CONCAT(sendToEntry.msg, '0');					(* message text*);
			END_IF;
		END_FOR
		(*sendToEntry.msg				:= Inputs;*)
		sendFifo.AddTail( new := sendToEntry );									(* add new entry to the send queue*)
		IF NOT sendFifo.bOk THEN												(* check for fifo overflow*)
			LogError( 'Send fifo overflow!', PLCPRJ_ERROR_SENDFIFO_OVERFLOW );
		END_IF
	END_IF

	(* send and receive messages *)
	fbPeerToPeer(
		sendFifo := sendFifo,
		receiveFifo := receiveFifo,
		sLocalHost := '',
		nLocalPort := 1002,
		bEnable := TRUE );
END_IF;

               %   ,     Ѕ§        
   SCODE_CODE мIL	мIL       шPP        >   FUNCTION SCODE_CODE : DWORD
VAR_INPUT
	sc		: UDINT;
END_VAR/   SCODE_CODE := 16#FFFF AND UDINT_TO_DWORD( sc );                 §џџџ, T U љR             TcpIp.lib 29.5.06 09:26:36 @М {D"   TcSystem.lib 9.3.10 10:21:30 @K!   TcBase.lib 14.5.09 11:14:08 @p_J"   STANDARD.LIB 5.6.98 11:03:02 @ж2x5      F_GetVersionTcpIp @      E_WinsockError       ST_SockAddr       ST_TcIpConnSvrResponse       ST_TcIpConnSvrUdpBuffer    	   T_HSOCKET                  FB_SocketAccept @          FB_SocketClose @          FB_SocketCloseAll @          FB_SocketConnect @          FB_SocketListen @          FB_SocketReceive @          FB_SocketSend @          FB_SocketUdpCreate @          FB_SocketUdpReceiveFrom @          FB_SocketUdpSendTo @             Global_Variables @       L   Р  ADSCLEAREVENTS @      E_IOAccessSize    
   E_OpenPath       E_SeekOrigin       E_TcEventClass       E_TcEventClearModes       E_TcEventPriority       E_TcEventStreamType       ExpressionResult       SFCActionType       SFCStepType    
   ST_AmsAddr       SYSTEMINFOTYPE       SYSTEMTASKINFOTYPE    
   T_AmsNetId       T_AmsNetIdArr    	   T_AmsPort    
   T_IPv4Addr       T_IPv4AddrArr       T_MaxString       TcEvent                   ADSLOGDINT @           ADSLOGEVENT @           ADSLOGLREAL @           ADSLOGSTR @           ADSRDDEVINFO @           ADSRDSTATE @           ADSRDWRT @           ADSRDWRTEX @           ADSRDWRTIND @           ADSRDWRTRES @           ADSREAD @           ADSREADEX @           ADSREADIND @           ADSREADRES @           ADSWRITE @           ADSWRITEIND @           ADSWRITERES @           ADSWRTCTL @           AnalyzeExpression @          AnalyzeExpressionCombined @          AnalyzeExpressionTable @          AppendErrorString @          CLEARBIT32 @           CSETBIT32 @           DRAND @           F_CompareFwVersion @          F_CreateAmsNetId @           F_CreateIPv4Addr @          F_GetVersionTcSystem @           F_IOPortRead @          F_IOPortWrite @          F_ScanAmsNetIds @          F_ScanIPv4AddrIds @          F_SplitPathName @          F_ToASC @          F_ToCHR @          FB_CreateDir @          FB_EOF @           FB_FileClose @           FB_FileDelete @           FB_FileGets @           FB_FileOpen @           FB_FilePuts @           FB_FileRead @           FB_FileRename @           FB_FileSeek @           FB_FileTell @           FB_FileWrite @           FB_PcWatchdog @          FB_RemoveDir @          FB_SimpleAdsLogEvent @          FILECLOSE @           FILEOPEN @           FILEREAD @           FILESEEK @           FILEWRITE @           FW_CallGenericFb @          FW_CallGenericFun @          GETBIT32 @           GETCPUACCOUNT @           GETCPUCOUNTER @           GETCURTASKINDEX @           GETSYSTEMTIME @           GETTASKTIME @          LPTSIGNAL @           MEMCMP @           MEMCPY @           MEMMOVE @           MEMSET @           ROL32 @           ROR32 @           SETBIT32 @           SFCActionControl @           SHL32 @           SHR32 @              Global_Variables @           z   FW_AdsClearEvents @      FW_NoOfByte       FW_SystemInfoType       FW_SystemTaskInfoType    
   FW_TcEvent                   FW_AdsLogDINT @           FW_AdsLogEvent @           FW_AdsLogLREAL @           FW_AdsLogSTR @           FW_AdsRdWrt @           FW_AdsRdWrtInd @           FW_AdsRdWrtRes @           FW_AdsRead @           FW_AdsReadDeviceInfo @           FW_AdsReadInd @           FW_AdsReadRes @           FW_AdsReadState @           FW_AdsWrite @           FW_AdsWriteControl @           FW_AdsWriteInd @           FW_AdsWriteRes @           FW_DRand @           FW_GetCpuAccount @           FW_GetCpuCounter @           FW_GetCurTaskIndex @           FW_GetSystemTime @           FW_GetVersionTcBase @           FW_LptSignal @           FW_MemCmp @           FW_MemCpy @           FW_MemMove @           FW_MemSet @           FW_PortRead @          FW_PortWrite @                 CONCAT @                	   CTD @        	   CTU @        
   CTUD @           DELETE @           F_TRIG @        
   FIND @           INSERT @        
   LEFT @        	   LEN @        	   MID @           R_TRIG @           REPLACE @           RIGHT @           RS @        
   SEMA @           SR @        	   TOF @        	   TON @           TP @              Global Variables 0 @                                             2                џџџџџџџџџџџџџџџџ  
             њџџџ  X?Ћ            јџџџ  р&ј&р'ј'                     POUs               Helper Functions                FB_Fifo                AddTail                 
   RemoveHead  "                      LogError  #                
   LogMessage  $                
   SCODE_CODE  %   џџџџ                FB_PeerToPeer  &                   MAIN      џџџџ           
   Data types                 E_ClientServerSteps  П                  ST_FifoEntry  Р   џџџџ             Visualizations  џџџџ              Global Variables                 Global_Variables                     TwinCAT_Configuration  Н                   Variable_Configuration  	   џџџџ                                                            М>L                         	   localhost            P      	   localhost            P      	   localhost            P          OJЏ