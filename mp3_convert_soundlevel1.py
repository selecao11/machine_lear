import os
#import glob 

#path1 = os.path.dirname(__file__) + "/folder1/" 
#print( path1 ) 

#list1 = glob.glob( path1 + "*.*" ) 
#print( list1 )  

file1="C:\\anaconda3\\envs\\machine_lear\\machine_lear\\convert_soundlevel1\\folder1\\20211214_202919.mp4"
file2="C:\\anaconda3\\envs\\machine_lear\\machine_lear\\convert_soundlevel1\\folder1\\done1\\20211214_202919_20dB.mp4"
ffmpeg_path="C:\\anaconda3\\envs\\machine_lear\\ffmpeg-master-latest-win64-gpl\\bin\\"
cmd1 = ffmpeg_path + "ffmpeg -i " + '"' + file1 + '" -filter:a "volume=20dB" "' + file2 + '"' 
print( cmd1 ) 
os.system( cmd1 ) 
