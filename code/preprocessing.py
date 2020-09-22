import pandas as pd

File_Path = "C:/Users/10559/Desktop/B/预处理/"
File_name_1 = ['附件三_异常值0(填充1).csv', '附件三_异常值0(填充2).csv']
File_name_2 = ['附件三_异常值0(填充1)1.csv', '附件三_异常值0(填充2)2.csv']
File_name_3 = ['附件三_异常值0(填充1)11.csv', '附件三_异常值0(填充2)22.csv']

# 判断3lamda，看元素是否满足条件
def Preprocess_lamda(filepath_r='', filepath_w=''):
    # 读取数据
    sheet_1 = pd.read_csv(filepath_r, skiprows= 0)
    n = 40
    lamda_1 = []
    average_1 = []

    for l in sheet_1:
        sum_ = 0.
        num_zero = 0
        
        # 查找零的个数
        for i in sheet_1[l]:
            if i == 0.:
                num_zero += 1

        # 均值为去除零值的均值
        if n > num_zero:
            average_ = sum(sheet_1[l])/(n-num_zero)
        else:
            average_ = 0
        # 加入到均值存放数组
        average_1.append(average_)

        for i in sheet_1[l]:
            # 零值为异常值，不予计算
            if i != 0.:
                sum_ += (i - average_) ** 2
        if n-num_zero > 1:
            tmp2 = (sum_/(n-1-num_zero)) ** 0.5
        else:
            tmp2 = 0.
        lamda_1.append(tmp2)

    sheet_11 = sheet_1
    for l in sheet_1:
        lam = lamda_1[int(l)-1] * 3
        ave = average_1[int(l)-1]
        for i in range(n):
            if sheet_1[l][i] != 0.:
                tmp = sheet_1[l][i] - ave
                if not -lam <= tmp <= lam:
                    sheet_11[l][i] = 0.
    sheet_11.to_csv(filepath_w)

# 补充缺失值
def Preprocess_zero(filepath_r='', filepath_w=''):
    sheet_1 = pd.read_csv(filepath_r)
    sheet_11 = sheet_1 
    for l in sheet_1:
        record_0 = []
        if 0. in sheet_1[l]:
            for i in range(len(sheet_1[l])):
                if sheet_1[l][i] == 0.:
                    record_0.append(i)
            if len(record_0) > len(sheet_1[l])/2:
                for i in range(len(sheet_1[l])):
                    if i not in record_0:
                        sheet_11[l][i] = 0.
            elif len(record_0):
                aver = sum(sheet_1[l])/(len(sheet_1[l])-len(record_0))
                # print(l, record_0, aver)
                for i in record_0:
                    sheet_11[l][i] = aver

    sheet_11.to_csv(filepath_w)

Preprocess_lamda(File_Path+File_name_1[0], File_Path+File_name_2[0])
Preprocess_lamda(File_Path+File_name_1[1], File_Path+File_name_2[1])
Preprocess_zero(File_Path+File_name_2[0], File_Path+File_name_3[0])
Preprocess_zero(File_Path+File_name_2[1], File_Path+File_name_3[1])