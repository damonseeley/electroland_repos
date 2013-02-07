@echo off
echo pausing for 2 mins before restart
PING 1.1.1.1 -n 1 -w 30000 >NUL
echo 1:30 remaining before restart
PING 1.1.1.1 -n 1 -w 30000 >NUL
echo 1:00 remaining before restart
PING 1.1.1.1 -n 1 -w 30000 >NUL
echo 30 secs remaining before restart
PING 1.1.1.1 -n 1 -w 30000 >NUL
echo restarting
start %1
exit
