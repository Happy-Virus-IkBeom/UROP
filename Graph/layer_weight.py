### level 7 => weight 21% remaining 시점에서 weight 를 0.5_2_3//1_1_1//1_1_2배로 조작하여 weight 가 결과에 어떠한 영향을 미치는지 보고자 함.

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Path와 level를 input으로 받아서 return 해주는 함수 => argparse 사용 parser.add_argument 사용해서 argument 받기
def path_Level():
    file_path = str(input("write the file location you want to read: "))
    return file_path


# FCN that change file name to logger.csv
def file_name_change_to_logger(file_path, i, x):

    full_path = str(f'{file_path}\\\\{x}\\{i}\main')
    file_names = os.listdir(f'{full_path}')

    # logger.csv 라는 확장자가 없으면 .csv를 뒤에 붙혀서 파일변환을 해줌.
    if not os.path.exists(f'{full_path}\logger.csv'):
        if "logger" in file_names:
            src = os.path.join(f'{full_path}', 'logger')
            dst = str("logger") + '.csv'
            dst = os.path.join(f'{full_path}', dst)
            os.rename(src,dst)

# extract loss value and make list having loss value.
def entire_loss_list(file_path, i, x):
    test_loss = []
    test_accuracy = []
    loss = []
    accuracy = []
    f = open(f'{file_path}\\\\{x}\\{i}\main\logger.csv')
    print(f"================  level : 7==  ================")
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
    file_path = path_Level()
    print(f'file path is: {file_path}')
    plt.cla()
    accuracy_list = []

    for x in ['0.5_2_3', '1_1_1', '1_1_2']:
        for i in [1, 2, 3]:
            print(f'{file_path}\\\\{x}\\{i}\main\logger.csv')
            file_name_change_to_logger(file_path, i, x)
            loss, length_of_test_loss, accuracy, length_of_test_accuracy = entire_loss_list(file_path, i, x)
            x_range = np.linspace(0, len(loss), len(loss))
            accuracy_list.append(accuracy)

        y2 = [(a + b + c) / 3 for a,b,c in zip (accuracy_list[0],accuracy_list[1],accuracy_list[2])] # 3번 시도해서 얻은 정확도를 3으로 나눠서 y2 리스트 생성. (mean for 3 trials)
        print(y2)
        plt.ylim(0.9775, 0.99)
        line = plt.plot(x_range, y2, label='accuracy')
        plt.setp(line, linewidth = 0.8 )
        y2 = []
        accuracy_list = []

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy at 21% remaining point')
    plt.legend(['x0.5,x2,x3','x1,x1,x1','x1,x1,x2'])
    plt.savefig(f'{file_path}\\accuracy.png')

plot_accuracy()