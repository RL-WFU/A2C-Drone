# A2C-Drone

A2RC and A3RC contain the A2C and A3C files respectively.
A2RC script:
python A2RC.py --seq_length 6 --num_steps 1000 --sight_dim 2 --num_episodes 5000 --lr 0.0001 --gamma 0.99 --extra_layer False --save_dir 'A2RC_saves' --layer_val 64 --net_config 0 --image 'image1.JPG'

A3RC script
python A3RC.py --seq_length 6 --num_steps 1000 --sight_dim 2 --num_episodes 5000 --lr 0.0001 --gamma 0.99 --extra_layer False --num_workers 8 --save_dir 'A3RCsaves' --layer_val 64 --net_config 0 --image 'image1.JPG'
