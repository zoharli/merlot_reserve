"""
Demo for doing interesting things with a video
"""
import sys
sys.path.append('../')
import os
import numpy as np

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp

# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

## First open the video and break it up into segments. you can only have 8.
# Each segment is 5 seconds so it corresponds to seconds 15 - 55 of the video


# Feel free to change the URL!
root_dir='/home/zhangheng/video_features/data'
for filename in os.listdir(root_dir):
    if not filename.endswith('.mp4'):
        continue
    try:
        all_video_segments = video_to_segments(os.path.join(root_dir,filename))
    except:
        continue
        

    result_arr=[]
    for start_idx in range(0,len(all_video_segments),8):
        print('======>start_idx:',start_idx)
        video_segments = all_video_segments[start_idx:start_idx+8]

        # Set up a fake classification task.
        video_segments[0]['text'] = 'in this video i\'ll be<|MASK|>'
        video_segments[0]['use_text_as_input'] = True
        for i in range(1,len(video_segments)):
            video_segments[i]['use_text_as_input'] = False

        video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=True)

        # Now we embed the entire video and extract the text. result is  [seq_len, H]. we extract a hidden state for every
        # MASK token
        out_h = model.embed_video(**video_pre)
        result_arr.append(np.array(out_h))
    result_arr=np.concatenate(result_arr)
    
    print('result_shape for '+filename+':',result_arr.shape)
    np.save('../tmp/'+filename.split('.')[0]+'_audio_visual_feat.npy',result_arr)
