#!/bin/bash

# ['PETS09_6.mp4', 'PETS09_7.mp4', 'PETS09-S2L2.mp4', 
# 'ETH_Jelmoli.mp4', '1_BEST.mp4', 'PETS09_1.mp4', 
# 'MOT16_07.mp4', '1_TEST.mp4', '4_BEST.mp4', 
# 'ETH_Bahnhof.mp4', '3_TEST.mp4', '5_BEST.mp4', 
# '5_TEST.mp4', 'Town_Center.avi', '3_BEST.mp4', 
# 'TUD_Stadtmitte.mp4', '4_TEST.mp4', 'ETH_Sunnyday.mp4', 
# 'PETS09_8.mp4', 'PETS09_0.mp4', 'TUD_Crossing.mp4', 
# '2_BEST.mp4', '2_TEST.mp4', 'PETS09_4.mp4', 
# 'PETS09_3.mp4', 'MOT16_08.mp4', 'PETS09_5.mp4',
# '1_Parking.avi', 'iccv07.mp4']

# FILE='retail_1.mp4'
FILE='PETS09_0.mp4'

python3 1_detection.py --test_video=$FILE

python3 2_generate_features.py --test_video=$FILE

# python3 3_deep_sort.py 
# python3 3_deep_sort.py --test_video=$FILE  --display=True
python3 3_deep_sort.py --test_video=$FILE

# python3 4_concatenate.py
python3 4_concatenate.py --test_video=$FILE

python3 5_show_results.py --test_video=$FILE --concat=True --smoothing=True --save_output=True 
# python3 5_show_results.py --test_video=$FILE

# python3 5_show_results.py
# python3 5_show_results.py --concat=True --smoothing=True 

# python3 5_show_results.py \
# --test_video=$FILE \
# --concat=True \
# --smoothing=True \
# --save_output=True \
# --save_fig=True \
# --is_plot=True

# generate submission file
# python3 6_submission.py 