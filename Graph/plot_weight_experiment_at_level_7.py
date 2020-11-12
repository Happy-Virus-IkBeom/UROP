### level 7 => weight 21% remaining 시점에서 weight 를 1 ~ 5배로 조작하여 weight 가 결과에 어떠한 영향을 미치는지 보고자 함.

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Path와 level를 input으로 받아서 return 해주는 함수 => argparse 사용 parser.add_argument 사용해서 argument 받기
def path_Level():
    file_path = str(input("write the file location you want to read: "))
    level = int(input("write the level you want to extract: "))
    return file_path ,level


# FCN that change file name to logger.csv
def file_name_change_to_logger(file_path, level, x):

    #full_path = str(f'{file_path}\\{x+1}배\\replicate_1\level_{level}\main')
    full_path = str(f'{file_path}\\\\replicate_{x+1}\level_{level}\main')
    file_names = os.listdir(f'{full_path}')

    # logger.csv 라는 확장자가 없으면 .csv를 뒤에 붙혀서 파일변환을 해줌.
    if not os.path.exists(f'{full_path}\logger.csv'):
        if "logger" in file_names:
            src = os.path.join(f'{full_path}', 'logger')
            dst = str("logger") + '.csv'
            dst = os.path.join(f'{full_path}', dst)
            os.rename(src,dst)

# extract loss value and make list having loss value.
def entire_loss_list(file_path, level, x):
    test_loss = []
    test_accuracy = []
    loss = []
    accuracy = []
    #f = open(f'{file_path}\\{x+1}배\\replicate_1\level_{level}\main\logger.csv')
    f = open(f'{file_path}\\\\replicate_{x+1}\level_{level}\main\logger.csv')
    print(f"================  level : {level}  ================")
    reader = csv.reader(f)
    for line in reader:
        if 'test_loss' in line:
            test_loss.append(line)
            # 아래에 test_loss 추가하는 부분 여기다가 추가. for문은 최소한으로 사용
        if 'test_accuracy' in line:
            test_accuracy.append(line)

    print("====================================================")
    f.close()
    test_loss = np.array(test_loss)
    length_of_test_loss = len(test_loss)
    test_accuracy = np.array(test_accuracy)
    length_of_test_accuracy = len(test_accuracy) # test_loss 와 length 같지만 보기쉽게 하나 더 추가함.

    # test_loss 추출하기
    for i in range (length_of_test_loss):
        loss_value = test_loss[i,2]
        loss_value = float(loss_value)
        loss.append(loss_value)

    # test_accuracy 추출하기
    for i in range (length_of_test_accuracy):
        accuracy_value = test_accuracy[i,2]
        accuracy_value = float(accuracy_value)
        accuracy.append(accuracy_value)

    return loss, length_of_test_loss, accuracy, length_of_test_accuracy

# Main Codebase
def plot_accuracy():

    # file_path is the file location that data is stored when you run the lottery.
    # level means pruning level where the data you want to extract is.
    file_path, level = path_Level()
    print(f'file path is: {file_path}')
    print(f'level is: {level}')
    plt.cla()
    i = 7

    for x in [6,7,8,9]:
        #print(f'{file_path}\\{x+1}배\\replicate_1\level_{i}\main\logger.csv')
        print(f'{file_path}\\\\replicate_{x+1}\level_{i}\main\logger.csv')
        file_name_change_to_logger(file_path, i, x)
        loss, length_of_test_loss, accuracy, length_of_test_accuracy = entire_loss_list(file_path, i, x)
        x_range = np.linspace(0, len(loss), len(loss))
        y1 = loss
        y2 = accuracy

        plt.ylim(0.9775, 0.99)
        line = plt.plot(x_range, y2, label='accuracy')
        plt.setp(line, linewidth = 0.8 )
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('accuracy at 21% remaining point')
    #plt.legend(['x1', 'x2', 'x3', 'x4', 'x5'])
    #plt.legend(['x0.5,x2,x3', 'x0.5,x3,x2', 'x2,x0.5,x3', 'x3,x0.5,x2', 'x2,x3,x0.5', 'x3,x2,x0.5','x1,x1,x1'])
    plt.legend(['x1,x1,x2','x1,x2,x1','x2,x1,x1','x1,x1,x1'])
    #plt.legend(['x3,x0.5,x2', 'x3,x2,x0.5'])
    plt.savefig(f'{file_path}\\accuracy3.png')

plot_accuracy()