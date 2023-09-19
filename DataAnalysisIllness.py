import pandas as pd # 匯入資料表處理套件
import matplotlib.pyplot as plt # 匯入視覺化套件

# 視覺化圖表中文顯示
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["font.size"] = 18

# 讀取類流感急診就診人次資料檔案
# 引用資料來源:政府資料開放平臺​-衛生福利部疾病管制署
df = pd.read_csv(r"C:\Users\hankh\Desktop\Python_Portfolios\Data_Analysis\RODS_Influenza_like_illness.csv")

# 檢查有無遺失值
print(df.isnull().any())

admin = list(set(df["縣市"])) # 縣市清單
year = sorted(list(set(df["年"]))) # 年份清單
week = sorted(list(set(df["週"]))) # 週數清單
# 年齡別清單
age = ['0-6', '7-12', '13-18', '19-64', '65+']

# 分析一 : 就診人次依年分別趨勢分析

# 資料表格處理
sumOfInfect = []
for i in admin:
    infectByAdmin = []
    for j in year[:-1]:
        infectNumber = sum(df["類流感急診就診人次"][(df["縣市"] == i) & (df["年"] == j)])
        infectByAdmin.append(infectNumber)
    sumOfInfect.append(infectByAdmin)
sumOfInfect2 = {}
for i, j in zip(admin, sumOfInfect):
    sumOfInfect2[i] = j
df2 = pd.DataFrame(sumOfInfect2)

# 使用折線圖觀察原始資料
ytitle = "總\n就\n診\n人\n次\n\n\n"
for i in range(len(admin)):
    plt.figure(figsize = (15, 5), dpi = 300)
    plt.plot(year[:-1], df2[admin[i]], "b-")
    plt.plot(year[:-1], df2[admin[i]], "ro")
    plt.title(admin[i])
    plt.xlabel("年分")
    plt.ylabel(ytitle, labelpad = 20, loc = "bottom", rotation = "horizontal")
    plt.xticks(year[:-1], year[:-1])
    plt.grid(True)
    plt.show()

# 在折線圖中發現2021年人次數特別低，判定受COVID-19疫情
# 影響，視為離群值而拔除，以避免後續分析因此受到太大影響
for i in range(len(year)):
    if year[i] == 2021:
        df2 = df2.drop(len(year) - 3)
        del year[i]
        break

# 將趨勢線從折線換成最小平方迴歸直線，觀察整體趨勢
import numpy as np
a = []
for i in year[:-1]:
    row = []
    for j in range(2):
        row.append((i - min(year) + 1) ** j)
    a.append(row)
a = np.array(a)
coefs = []
for i in range(len(df2.columns)):
    b = np.array(df2[df2.columns[i]])
    x = np.linalg.inv(a.T @ a) @ (a.T @ b)
    coefs.append(x)
predicts = []
for p in range(len(df2.columns)):
    predict = []
    for i in year[:-1]:
        predict_value = 0
        for j, k in enumerate(coefs[p]):
            predict_value += ((i - min(year)) ** j) * k
        predict.append(predict_value)
    predicts.append([predict, (predict[-1] - predict[0]) / len(predict)]) # 計算平均增加率
for i in range(len(df2.columns)):
    plt.figure(figsize = (15, 5), dpi = 300)
    plt.plot(year[:-1], df2[admin[i]], "ro")
    plt.plot(year[:-1], predicts[i][0], "b-")
    plt.title(admin[i] + " (平均每年增加約{}人)".format(int(predicts[i][1])))
    plt.xlabel("年分")
    plt.ylabel(ytitle, labelpad = 20, loc = "bottom", rotation = "horizontal")
    plt.xticks(year[:-1], year[:-1])
    plt.grid(True)
    plt.show()



# 分析二 : 就診人次按週別分析
 
# 把平均增加率大於零的縣市再整理出一個名單(名單2)
admin2 = []
for i in range(len(predicts)):
    if predicts[i][1] > 0:
        admin2.append(df2.columns[i])

# 資料表格處理
sumOfInfectByWeek = []
for i in admin2:
    infectByAdmin = []
    for j in week:
        infectNumber = sum(df["類流感急診就診人次"][(df["縣市"] == i) & (df["週"] == j) & (df["年"] != 2021) & (df["年"] != 2023)]) / len(year[:-1])
        infectByAdmin.append(infectNumber)
    sumOfInfectByWeek.append(infectByAdmin)
sumOfInfectByWeek2 = {}
for i, j in zip(admin2, sumOfInfectByWeek):
    sumOfInfectByWeek2[i] = j
df3 = pd.DataFrame(sumOfInfectByWeek2)

# 使用長條圖(週別作為離散/類別資料)觀察資料
ytitle = "平\n均\n就\n診\n人\n次\n\n\n\n\n"
means = {}
modes = []
for i in range(len(admin2)):
    means[admin2[i]] = np.mean(df3[admin2[i]])
    mode = []
    plt.figure(figsize = (10, 6), dpi = 300)
    plt.bar(week, df3[admin2[i]])
    plt.axhline(y = means[admin2[i]], color = "red", ls = "--")
    plt.title(admin2[i] + " (紅線全年平均值: {:.2f})".format(means[admin2[i]]))
    for j, k in enumerate(df3[admin2[i]]):
        if k > means[admin2[i]]:
            mode.append(j)
            plt.text(x = week[j] - 0.5,
                      y = df3[admin2[i]][j] + 0.8,
                      s = j + 1,
                      fontsize = 11)
    plt.xlabel("週數(第1週至第53週)")
    plt.ylabel(ytitle, labelpad = 20, loc = "bottom", rotation = "horizontal")
    plt.grid(True)
    plt.show()
    modes.append(mode)

# 名單2內縣市在第1週至第53週之各週平均
# 做超過總平均(第1週至第53週)次數累計後
# 再使用長條圖觀察次數分布
modeCount = [0 for i in week]
for i in modes:
    for j in i:
        for k in week:
            if j == k:
                modeCount[k - 1] += 1
plt.figure(figsize = (10, 6), dpi = 300)
plt.bar(week, modeCount)
plt.title("各週超過平均值次數長條圖")
plt.xlabel("週")
plt.ylabel("次\n數\n\n\n\n\n\n", labelpad = 20, loc = "bottom", rotation = "horizontal")
plt.show()

print(sum(modeCount[:25]) / sum(modeCount))
# 印出前25週在總共53週的占比


# 分析三 : 就診人次按年齡別分析

# 資料表格處理
ageMean = []
sumOfInfectByAge = []
for i in admin2:
    infectByAdmin = []
    for j in age:
        infectNumber = sum(df["類流感急診就診人次"][(df["縣市"] == i) & (df["年齡別"] == j)])
        infectByAdmin.append(infectNumber)
    sumOfInfectByAge.append(infectByAdmin)
for i in sumOfInfectByAge:
    ageMean.append(np.mean(i))
sumOfInfectByAge2 = {}
for i, j in zip(admin2, sumOfInfectByAge):
    sumOfInfectByAge2[i] = j
df4 = pd.DataFrame(sumOfInfectByAge2)

# 使用長條圖觀察資料
ytitle = "累\n計\n就\n診\n人\n次\n\n"
for i in range(len(admin2)):
    plt.figure(dpi = 300)
    plt.bar(age, df4[admin2[i]])
    plt.title(admin2[i] + " (平均人次: {:.2f})".format(ageMean[i]))
    plt.xlabel("年齡別")
    plt.ylabel(ytitle, labelpad = 20, loc = "bottom", rotation = "horizontal")
    plt.axhline(y = ageMean[i], color = "red", ls = "--")
    plt.grid(True)
    plt.show()

# 累計"各年齡別累計人次"中超過"總人次平均"之次數   
# 再行繪製長條圖判定重點為哪些年齡別
counts = [0 for i in range(5)]
for i, j in enumerate(sumOfInfectByAge):
    for k in range(len(j)):
        if j[k] > ageMean[i]:
            counts[k] += 1
plt.figure(figsize = (10, 6), dpi = 300)
plt.bar(age, counts)
plt.title("年齡超過平均值次數長條圖")
plt.xlabel("年齡別")
plt.ylabel("次\n數\n\n\n\n\n\n", labelpad = 20, loc = "bottom", rotation = "horizontal")
plt.show()