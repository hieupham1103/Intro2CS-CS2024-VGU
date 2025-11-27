export CUDA_VISIBLE_DEVICES=0
model_type=yolo
type=RGB
mode_path=rgb

python evaluate.py --model-path ./checkpoints/${mode_path}.pt --model-type ${model_type} --ood-dir ${type}_TEST_OD --video-type ${type} --ood-test