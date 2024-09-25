import os

folder_path = '/data/lh/ISTFormer/datasets/CD-188'
num = 1

if __name__ == '__main__':
    for file in os.listdir(folder_path):
        s = '%03d' % num  # 前面补零占位
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, 'coal_' + str(s) +'.jpg'))
        num += 1
