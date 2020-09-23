import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Path와 level를 input으로 만들어서 return 해주는 함수
def Path_Level():
    file_path = str(input("write the file location you want to read: "))
    level = int(input("write the level you want to extract: "))
    return file_path ,level

# FCN that change file name to logger.csv
def file_name_change_to_logger(file_path, level):

    full_path = str(f'{file_path}\\replicate_1\level_{level}\main')
    file_names = os.listdir(f'{full_path}')

    # logger.csv 라는 확장자가 없으면 .csv를 뒤에 붙혀서 파일변환을 해줌.
    if not os.path.exists(f'{full_path}\logger.csv'):
        if "logger" in file_names:
            src = os.path.join(f'{full_path}', 'logger')
            dst = str("logger") + '.csv'
            dst = os.path.join(f'{full_path}', dst)
            os.rename(src,dst)

# FCN that read csv file in file_path (가장 상위 directory로 입력하기) // It is needed when you check or modify the logger.csv file
def read_csv(file_path,level):
    f = open(f'{file_path}\\replicate_1\level_{level}\main\logger.csv')
    print(f"================  level : {level}  ================")
    reader = csv.reader(f)
    for line in reader:
        print(line)
    print("====================================================")
    f.close()

# extract loss value and make list having loss value.
def entire_loss_list(file_path, level):
    test_loss = []
    loss = []
    f = open(f'{file_path}\\replicate_1\level_{level}\main\logger.csv')
    print(f"================  level : {level}  ================")
    reader = csv.reader(f)
    for line in reader:
        if 'test_loss' in line:
            test_loss.append(line)
    print("====================================================")
    f.close()
    test_loss = np.array(test_loss)
    length_of_test_loss = len(test_loss)

    # test_loss 추출하기
    for i in range (length_of_test_loss):
        loss_value = test_loss[i,2]
        loss_value = float(loss_value)
        loss.append(loss_value)

    return loss, length_of_test_loss

def Index_Early_Stopping_Accuracy(loss, length_of_test_loss, level):
    Index = 0
    for i in range (length_of_test_loss):
        if loss[i] >= loss[i+1]:
            Index += 1

        else:
            break

    return Index

# Main Codebase
def Early_Stopping_Loss_List():

    # file_path is the file location that data is stored when you run the lottery.
    # level means pruning level where the data you want to extract is.
    file_path, level = Path_Level()
    print(f'file path is: {file_path}')
    print(f'level is: {level}')

    Loss_list = []
    Index_list = []
    for i in range(level + 1):
        print(f'{file_path}\\replicate_1\level_{i}\main\logger.csv')
        file_name_change_to_logger(file_path, i)
        #read_csv(file_path, i)
        loss, length_of_test_loss = entire_loss_list(file_path, i)
        print(loss)
        Index = Index_Early_Stopping_Accuracy(loss, length_of_test_loss, i)
        #print(Index)
        loss = loss[Index]
        Loss_list.append(loss)
        Index_list.append(Index)
        print(f'Loss_list is: \n {Loss_list}')
        print(f'Index_list is: \n {Index_list}')

    return Loss_list, Index_list, level

def Plot_Early_Stopping_Loss():
    Loss_list, Index_list, level, file_path = Early_Stopping_Loss_List()
    x_range = np.linspace(0, level, level + 1)
    y1 = Loss_list
    y2 = Index_list
    plt.cla()
    plt.subplot(2,1,1)
    plt.plot(x_range, y1, color = 'blue', marker = '+', label = 'Early_Stopping_Loss')
    plt.subplot(2,1,2)
    plt.plot(x_range, y2, color = 'red', marker = 'x', label = 'Early_Stopping_epoch')
    plt.savefig(f'{file_path}\Early_Stopping_Loss&Epoch.png')

Plot_Early_Stopping_Loss()
