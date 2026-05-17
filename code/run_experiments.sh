#!/bin/bash

# 创建一个专门存放日志的文件夹
mkdir -p global_logs

echo "开始实验全流程排队..."
start_time=$(date +%Y-%m-%d\ %H:%M:%S)
echo "总任务开始时间: $start_time"

# --- 1. 全监督实验 (Fully Supervised) ---
# echo "Running Task 1/12: SwinUNet Fully Supervised..."
# python train_fully_supervised_2D_ViT.py --root_path ../data/ACDC --exp ACDC/swinunet --model swinunet --max_iterations 10000 --batch_size 24 --num_classes 4 > global_logs/fully_swin.log 2>&1

# echo "Running Task 2/12: VIM (MambaUNet)..."
# python train_fully_supervised_2D_VIM.py --root_path ../data/ACDC --exp ACDC/VIM --model mambaunet --max_iterations 10000 --batch_size 24 --num_classes 4 > global_logs/fully_vim.log 2>&1

# --- 2. 半监督 Mamba UNet 实验 (Semi-supervised) ---
# echo "Running Task 3/12: Semi Mamba UNet (3 labeled)..."
# python train_Semi_Mamba_UNet.py --root_path ../data/ACDC --exp ACDC/Semi_Mamba_UNet_L3 --max_iterations 30000 --labeled_num 3 --batch_size 16 --num_classes 4 > global_logs/semi_mamba_l3.log 2>&1

# echo "Running Task 4/12: Semi Mamba UNet (7 labeled)..."
# python train_Semi_Mamba_UNet.py --root_path ../data/ACDC --exp ACDC/Semi_Mamba_UNet_L7 --max_iterations 30000 --labeled_num 7 --batch_size 16 --num_classes 4 > global_logs/semi_mamba_l7.log 2>&1

# --- 3. Mean Teacher 实验 ---
echo "Running Task 5-8: Mean Teacher series..."
# python train_mean_teacher_2D.py --root_path ../data/ACDC --model unet --exp ACDC/Mean_Teacher_L3 --max_iterations 30000 --labeled_num 3 --batch_size 16 --num_classes 4 > global_logs/mt_unet_l3.log 2>&1
python train_mean_teacher_ViT.py --root_path ../data/ACDC --model swinunet --exp ACDC/Mean_Teacher_ViT_L3 --max_iterations 30000 --labeled_num 3 --batch_size 16 --num_classes 4 > global_logs/mt_vit_l3.log 2>&1
# python train_mean_teacher_2D.py --root_path ../data/ACDC --model unet --exp ACDC/Mean_Teacher_L7 --max_iterations 30000 --labeled_num 7 --batch_size 16 --num_classes 4 > global_logs/mt_unet_l7.log 2>&1
python train_mean_teacher_ViT.py --root_path ../data/ACDC --model swinunet --exp ACDC/Mean_Teacher_ViT_L7 --max_iterations 30000 --labeled_num 7 --batch_size 16 --num_classes 4 > global_logs/mt_vit_l7.log 2>&1

# --- 4. Uncertainty Aware Mean Teacher 实验 ---
echo "Running Task 9-12: Uncertainty Aware series..."
# python train_uncertainty_aware_mean_teacher_2D.py --root_path ../data/ACDC --model unet --exp ACDC/UA_MT_L3 --max_iterations 30000 --labeled_num 3 --batch_size 16 --num_classes 4 > global_logs/uamt_unet_l3.log 2>&1
python train_uncertainty_aware_mean_teacher_2D_ViT.py --root_path ../data/ACDC --model swinunet --exp ACDC/UA_MT_ViT_L3 --max_iterations 30000 --labeled_num 3 --batch_size 16 --num_classes 4 > global_logs/uamt_vit_l3.log 2>&1
# python train_uncertainty_aware_mean_teacher_2D.py --root_path ../data/ACDC --model unet --exp ACDC/UA_MT_L7 --max_iterations 30000 --labeled_num 7 --batch_size 16 --num_classes 4 > global_logs/uamt_unet_l7.log 2>&1
python train_uncertainty_aware_mean_teacher_2D_ViT.py --root_path ../data/ACDC --model swinunet --exp ACDC/UA_MT_ViT_L7 --max_iterations 30000 --labeled_num 7 --batch_size 16 --num_classes 4 > global_logs/uamt_vit_l7.log 2>&1

# --- 5. 测试环节 ---
# 注意：你需要把 xxx 替换为你真正想要测试的模型路径
# echo "Running Evaluation..."
# python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/swinunet --model swinunet

end_time=$(date +%Y-%m-%d\ %H:%M:%S)
echo "所有任务已完成！"
echo "开始时间: $start_time"
echo "结束时间: $end_time"