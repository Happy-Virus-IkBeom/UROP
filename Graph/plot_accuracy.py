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
    test_accuracy = []
    loss = []
    accuracy = []
    f = open(f'{file_path}\\replicate_1\level_{level}\main\logger.csv')
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
    #Draw a graph at level with 100%, 51%, 21% of the original amount remaining.
    for i in [0, 3, 7]:
        print(f'{file_path}\\replicate_1\level_{i}\main\logger.csv')
        file_name_change_to_logger(file_path, i)
        loss, length_of_test_loss, accuracy, length_of_test_accuracy = entire_loss_list(file_path, i)
        x_range = np.linspace(0, len(loss), len(loss))
        y1 = loss
        y2 = accuracy

        plt.ylim(0.94, 0.99)
        line = plt.plot(x_range, y2, label='accuracy')
        plt.setp(line, linewidth = 0.8 )
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('accuracy')
        plt.legend(['0', '3', '7'])
        plt.savefig(f'{file_path}\\accuracy1.png')

    plt.cla()
    # Draw a graph at level with 7%, 3.6%, 1.9% of the original amount remaining.
    for i in [12, 15, 18]:
        print(f'{file_path}\\replicate_1\level_{i}\main\logger.csv')
        file_name_change_to_logger(file_path, i)
        loss, length_of_test_loss, accuracy, length_of_test_accuracy = entire_loss_list(file_path, i)
        x_range = np.linspace(0, len(loss), len(loss))
        y1 = loss
        y2 = accuracy

        plt.ylim(0.94,0.99)
        line = plt.plot(x_range, y2, label='accuracy')
        plt.setp(line, linewidth = 0.8 )
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('accuracy')
        plt.legend(['12', '15', '18'])
        plt.savefig(f'{file_path}\\accuracy2.png')

# Main Codebase
def plot_loss():

    # file_path is the file location that data is stored when you run the lottery.
    # level means pruning level where the data you want to extract is.
    file_path, level = path_Level()
    print(f'file path is: {file_path}')
    print(f'level is: {level}')
    plt.cla()
    #Draw a graph at level with 100%, 51%, 21% of the original amount remaining.
    for i in [0, 3, 7]:
        print(f'{file_path}\\replicate_1\level_{i}\main\logger.csv')
        file_name_change_to_logger(file_path, i)
        loss, length_of_test_loss, accuracy, length_of_test_accuracy = entire_loss_list(file_path, i)
        x_range = np.linspace(0, len(loss), len(loss))
        y1 = loss
        y2 = accuracy

        plt.plot(x_range, y1, label='loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss')
        plt.legend(['0', '3', '7'])
        plt.savefig(f'{file_path}\loss1.png')


    plt.cla()
    # Draw a graph at level with 7%, 3.6%, 1.9% of the original amount remaining.
    for i in [12, 15, 18]:
        print(f'{file_path}\\replicate_1\level_{i}\main\logger.csv')
        file_name_change_to_logger(file_path, i)
        loss, length_of_test_loss, accuracy, length_of_test_accuracy = entire_loss_list(file_path, i)
        x_range = np.linspace(0, len(loss), len(loss))
        y1 = loss
        y2 = accuracy

        plt.plot(x_range, y1, label='loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss')
        plt.legend(['12', '15', '18'])
        plt.savefig(f'{file_path}\loss2.png')


plot_accuracy()
#plot_loss()