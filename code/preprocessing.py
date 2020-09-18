import pandas as pd

File_Path = "C:/Users/10559/Desktop/B/预处理/"
# File_name = os.listdir(File_Path)
File_name_r = ['附件三_异常值0(填充1).csv', '附件三_异常值0(填充2).csv']
File_name_w = ['附件三_异常值0(填充1)1.csv', '附件三_异常值0(填充2)2.csv']

sheet_1 = pd.read_csv(File_Path + File_name_r[0], skiprows= 0)
sheet_2 = pd.read_csv(File_Path + File_name_r[1], skiprows= 0)

n = 40
lambad_1 = []
average_1 = []
lambad_2 = []
average_2 = []

for l in sheet_1:
    sum_ = 0.
    average_ = sum(sheet_1[l])/n
    average_1.append(average_)
    for i in sheet_1[l]:
        sum_ += (i - average_) ** 2
    tmp2 = (sum_/(n-1))**0.5
    lambad_1.append(tmp2)

for l in sheet_2:
    sum_ = 0.
    average_ = sum(sheet_2[l])/n
    average_2.append(average_)
    for i in sheet_2[l]:
        sum_ += (i - average_) ** 2
    tmp2 = (sum_/(n-1))**0.5
    lambad_2.append(tmp2)

k = 0
sheet_11 = sheet_1
for l in sheet_1:
    lam = lambad_1[int(l)-1] * 3
    ave = average_1[int(l)-1]
    
    for i in range(n):
        tmp = sheet_1[l][i] - ave
        if not -lam <= tmp <= lam:
            k += 1
            sheet_11[l][i] = 0.
            print(l,lam,sheet_1[l][i])
print(k)
sheet_11.to_csv(File_Path + File_name_w[0])

k = 0
sheet_22 = sheet_2
for l in sheet_2:
    lam = lambad_2[int(l)-1] * 3
    ave = average_2[int(l)-1]
    for i in range(n):
        tmp = sheet_2[l][i] - ave
        if not -lam <= tmp <= lam:
            k += 1
            sheet_22[l][i] = 0.
            # print(l,lam,sheet_2[l][i] - ave)
print(k)
sheet_22.to_csv(File_Path + File_name_w[1])