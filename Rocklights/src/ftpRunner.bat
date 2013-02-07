@echo off
>  C:\tmpFtpScript.txt echo open electroland.net
>> C:\tmpFtpScript.txt echo rocklights
>> C:\tmpFtpScript.txt echo l0gcheck
>> C:\tmpFtpScript.txt echo put %1
>> C:\tmpFtpScript.txt echo bye
ftp -i -s:C:\tmpFtpScript.txt 
del C:\tmpFtpScript.txt 
exit