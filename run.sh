# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# pip install huggingface-hub==0.25.0
# pip install peft==0.13.0


# CUDA_VISIBLE_DEVICES=4 python inf_folder.py --relight_type golden_sunlight_1 --image_width 512 --image_height 512 &
# CUDA_VISIBLE_DEVICES=5 python inf_folder.py --relight_type noon_sunlight_1 --image_width 512 --image_height 512 &
# CUDA_VISIBLE_DEVICES=6 python inf_folder.py --relight_type foggy_1 --image_width 512 --image_height 512 &
# CUDA_VISIBLE_DEVICES=7 python inf_folder.py --relight_type moonlight_1 --image_width 512 --image_height 512 &
# wait

# python inf_folder.py --relight_type golden_sunlight_1 \
#     --input_dir /home/shenzhen/Datasets/depth/workzone_segm/boston/image \
#     --subdir depth/workzone_segm/boston \
#     --center_crop \

# CUDA_VISIBLE_DEVICES=1 python inf_folder.py --relight_type foggy_1 \
#     --input_dir /home/shenzhen/Datasets/depth/workzone_segm/boston/image \
#     --subdir depth/workzone_segm/boston \
#     --center_crop


relight_types=(golden_sunlight_1 foggy_1)
folders=(/scratch/shenzhen/Datasets/ROADWork/extracted_frames/boston_*)

for relight_type in "${relight_types[@]}"; do
    echo "Starting $relight_type ..."

    for gpu in 0 1 2 3 4 5 6 7; do
    (
        for ((i=gpu; i<${#folders[@]}; i+=8)); do
            folder="${folders[$i]}"
            folder_name=$(basename "$folder")
            subdir="ROADWork/extracted_frames/$folder_name"

            CUDA_VISIBLE_DEVICES=$gpu python inf_folder.py \
                --relight_type "$relight_type" \
                --input_dir "$folder/image" \
                --subdir "$subdir" \
                --center_crop
        done
    ) &
    done

    wait
    echo "Finished $relight_type"
done