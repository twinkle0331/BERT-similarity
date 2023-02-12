import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    layers=[]
    layer_num=13
    for i in range(layer_num):
        layers.append([])

    for seed_1 in range(25):
        for seed_2 in range(seed_1+1):
            if seed_1 == seed_2:
                continue
            print(seed_1,seed_2)
            input_dir = f"/home/ubuntu//trinkle/probing/output/caillen/similarity/mnli/mode_6/seed_{seed_1}_{seed_2}.npy"
            seq_no_res = np.diagonal(np.load(os.path.join(input_dir)))
            for i in range(len(seq_no_res)):
                layers[i].append(seq_no_res[i])

    box=[]
    for i in range(layer_num):
        box.append(layers[i])

    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    #plt.title('pretraining: mutual similarity of different seeds', fontsize=20)  # 标题，并设定字号大小
    plt.title('fine-tuned: mutual seq_no_res similarity of different seeds', fontsize=20)

    labels = [] # 图例
    for i in range(layer_num):
        labels.append(i)

    plt.boxplot(box, labels=labels,whis=(5,95))  # grid=False：代表不显示背景中的网格线
    # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
    plt.show()  # 显示图像

if __name__ == "__main__":

    main()