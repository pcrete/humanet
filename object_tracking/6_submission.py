import os

cap_dir = '../dataset/captures'

video_list = os.listdir(cap_dir)
video_list.sort()

with open('../dataset/20p32w0047.txt', 'w') as f:
	for video_name in video_list:

		N = len(os.listdir(os.path.join(cap_dir, video_name)))
		
		f.write(video_name+': '+str(N)+'\n')