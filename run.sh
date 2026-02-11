# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# pip install huggingface-hub==0.25.0
# pip install peft==0.13.0

CUDA_VISIBLE_DEVICES=4 python inf_folder.py --relight_type noon_sunlight_1 --image_width 512 --image_height 512 &
CUDA_VISIBLE_DEVICES=5 python inf_folder.py --relight_type golden_sunlight_1 --image_width 512 --image_height 512 &
CUDA_VISIBLE_DEVICES=6 python inf_folder.py --relight_type foggy_1 --image_width 512 --image_height 512 &
CUDA_VISIBLE_DEVICES=7 python inf_folder.py --relight_type moonlight_1 --image_width 512 --image_height 512 &
wait