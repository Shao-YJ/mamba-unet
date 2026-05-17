import os
import pandas as pd
import re

def extract_metrics(base_path):
    results = []
    # 遍历 ACDC 结果文件夹
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        for model in os.listdir(folder_path):
            log_path = os.path.join(folder_path, model)
            # 寻找性能记录文件，通常名为 val_performance.txt 或类似
            perf_file = os.path.join(log_path, 'log.txt') 
        
            print(perf_file)
            if os.path.exists(perf_file):
                with open(perf_file, 'r') as f:
                    content = f.read()
                    # 使用正则抓取 Dice 指标 (假设格式为 Mean_Dice: 0.xxxx)
                    dice = re.findall(r'mean_dice : (0\.\d+)', content)
                    hd95 = re.findall(r'mean_hd95 : (\d+\.\d+)', content)
                    
                    results.append({
                        "Experiment": folder,
                        "Dice": dice[-1] if dice else "N/A",
                        "HD95": hd95[-1] if hd95 else "N/A"
                    })
            else:
                print("No such file")
    
    df = pd.DataFrame(results)
    df.to_csv("experiment_summary.csv", index=False)
    print("✅ 所有实验结果已汇总至 experiment_summary.csv")

# 使用时指向你的模型输出根目录
extract_metrics('../../model/ACDC')