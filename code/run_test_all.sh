#!/bin/bash

# 创建测试结果日志文件夹
mkdir -p global_logs/test_results

echo "🚀 开始全流程模型测试 (已适配实际路径)..."

# --- 0. 基础全监督 UNet (手动确认你的路径名，假设是 unet) ---
echo "Testing Task 0: Standard UNet..."
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/unet_140_labeled --model unet > global_logs/test_results/test_fully_unet.log 2>&1

# --- 1. 全监督对比 (SwinUNet & VIM) ---
echo "Testing Task 1: SwinUNet Fully..."
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/swinunet_140_labeled --model swinunet > global_logs/test_results/test_fully_swin.log 2>&1

echo "Testing Task 2: VIM (MambaUNet) Fully..."
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/VIM_140_labeled --model mambaunet > global_logs/test_results/test_fully_vim.log 2>&1

# --- 2. 半监督 Mamba UNet ---
echo "Testing Task 3&4: Semi Mamba UNet..."
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Semi_Mamba_UNet_L3_3 --model mambaunet > global_logs/test_results/test_semi_mamba_l3.log 2>&1
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Semi_Mamba_UNet_L7_7 --model mambaunet > global_logs/test_results/test_semi_mamba_l7.log 2>&1

# --- 3. Mean Teacher 系列 (CNN & ViT) ---
echo "Testing Task 5-8: Mean Teacher series..."
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher_L3_3_labeled --model unet > global_logs/test_results/test_mt_unet_l3.log 2>&1
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher_L7_7_labeled --model unet > global_logs/test_results/test_mt_unet_l7.log 2>&1
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher_ViT_L3_3_labeled --model swinunet > global_logs/test_results/test_mt_vit_l3.log 2>&1
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher_ViT_L7_7_labeled --model swinunet > global_logs/test_results/test_mt_vit_l7.log 2>&1

# --- 4. Uncertainty Aware Mean Teacher 系列 (CNN & ViT) ---
echo "Testing Task 9-12: UA-MT series..."
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/UA_MT_L3_3_labeled --model unet > global_logs/test_results/test_uamt_unet_l3.log 2>&1
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/UA_MT_L7_7_labeled --model unet > global_logs/test_results/test_uamt_unet_l7.log 2>&1
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/UA_MT_ViT_L3_3_labeled --model swinunet > global_logs/test_results/test_uamt_vit_l3.log 2>&1
python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/UA_MT_ViT_L7_7_labeled --model swinunet > global_logs/test_results/test_uamt_vit_l7.log 2>&1

echo "------------------------------------------------"
echo "✅ 所有测试已完成。查看简报："
grep "Mean_Dice" global_logs/test_results/*.log