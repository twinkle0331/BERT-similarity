import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker


plt.rcParams['font.size'] = 14

def main():
    layers=[]
    layer_num=13
    for i in range(layer_num):
        layers.append([])

    for seed_1 in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
        for seed_2 in [5, 6, 7, 8]:
            input_dir = f"/home/ubuntu/trinkle/probing/output/caillen/similarity/mnli/mode_6/seed_{seed_1}_{seed_2}.npy"
            seq_no_res = np.diagonal(np.load(os.path.join(input_dir)))
            for i in range(len(seq_no_res)):
                layers[i].append(seq_no_res[i])

    box=[]
    for i in range(layer_num):
        box.append(layers[i])

    fig = plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    ax = fig.add_subplot(111)
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Squared Error")
    plt.ylim(ymax=0.15)
    #plt.title('pretraining: mutual similarity of different seeds', fontsize=20)  # 标题，并设定字号大小

    labels = [] # 图例
    for i in range(layer_num):
        labels.append(i)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    plt.boxplot(box, labels=labels, whis=(5, 95),
                flierprops=dict(markersize=4),
                whiskerprops=dict(linestyle='--'))

    output_dir = r'/home/ubuntu/trinkle/probing/output/caillen/similarity_figure'
    file = f"{output_dir}/box_figure.pdf"
    plt.savefig(file, figsize=(3, 2), dpi=900)
    plt.show()  # 显示图像

if __name__ == "__main__":

    main()