conda activate pps
python ./pps-main/train.py --config-path ./pps-main/config/activitynet/config_refact.json --ckpt-path ./pps-main/checkpoints/activitynet/model_refact.pt --eval
