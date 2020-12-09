start docker_command.bat
sleep 3
grep '     or http://' ./docker_logs.txt > ./jupyter_path.txt
sed -i 's/     or //g' ./jupyter_path.txt
var=`cat jupyter_path.txt`
echo $var
start "" "C:\Program Files\Mozilla Firefox\firefox.exe" $var