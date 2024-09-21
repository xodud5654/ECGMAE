import glob2
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

save_path = "SAVE_PATH"
csv_path = "CSV_FILE_PATH"
image_path = "/media/airl2/hdd1/SSL_ECG/data/gray"
dtype = "test"
diseases = ["AF","AFIB","SA","SB","SR","ST","SVT"]

# ALL lead stack image
for disease in diseases:
    path = f"{image_path}/lead_1/{dtype}/{disease}"
    for p in glob2.glob(f"{path}/*.png"):
        csv = "_".join(p.split("_")[3:-1])
        data = pd.read_csv(f"{csv_path}/{csv}.csv",header=None)
        plt.figure(figsize=(6,6))
        plt.style.use('dark_background')

        for i in range(1,13,1):
            plt.subplot(12,1,i)
            lead_data = data.iloc[:,i-1]
            plt.plot(lead_data,color="white",linewidth=2.0)
            plt.xlim(0, len(lead_data))  # x축 범위 설정
            plt.ylim(min(lead_data) - 5, max(lead_data) + 5)  # y축 범위 설정
            plt.axis("off")
        save_p = f"{save_path}/{dtype}/{disease}"
        os.makedirs(save_p,exist_ok=True)
        plt.savefig(f"{save_p}/{csv}.png",bbox_inches="tight", pad_inches = 0,dpi=50)
        plt.close()