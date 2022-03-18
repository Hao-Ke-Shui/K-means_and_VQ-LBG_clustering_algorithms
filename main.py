import numpy as np
import matplotlib.pyplot as plt
import time


# 可以是高斯分布生成数据
# def func1(样本数, 类别数，类别中心):
def Databuild(sample_num, cla_num, *cla_locat):
    cla = int(sample_num/cla_num)  # 每个类别有多少个点
    data = np.random.seed(1)
    data = np.zeros((cla_num,cla,2))
    # np.random.seed(1)

    for i in range(cla_num):
        data[i] = np.random.multivariate_normal(list(cla_locat)[0][i],[[2,0],[0,2]],cla )
    return data

# 输出码书以及各个码字对应的样本点
def VQ_LBG(code_num, para1 ,data):
    # 初始化参数
    code = list()
    code_star = np.zeros((code_num, 2))
    code_star_sample = list()
    itera_num = np.log2(code_num) # 迭代次数
    data1 = np.zeros((data.shape[0], 2)) # 第二大步计算方差
    data2 = np.zeros((code_num, 2)) # 循环中计算方差
    data_min = np.zeros((code_num)) # 循环中储存方差最小值

    # 第二大步
    code_star[0] = [data[:,0].sum()/data.shape[0], data[:,1].sum()/data.shape[0]]
    for i in range(data.shape[0]):
        data1[i] = data[i] - code_star[0]
    data1 = data1**2
    variance_star = data1.sum()/data.shape[0]
    variance_i = variance_star

    for i in range(int(itera_num)):
        # 第三大步  分裂
        print('**********************************第 '+str(i+1)+' 次分裂***************************************')
        for j in range(2**i):
            code_star[j] = (1 + para1) * code_star[j]
            code_star[2**i + j] = (1 - para1) * code_star[j]
        # print(code_star)
        #第四大步  当前有 2**(i+1) 个码字
        variance = variance_star
        variance_star = 0
        iii = 0
        iiiii = 0
        while (variance - variance_star)/variance > para1 :
            if variance_star != 0:
                # 每次循环更新方差值
                variance = variance_star

            print('--------------第 '+str(i+1)+' 次分裂的第 '+str(iii)+' 次更新码字位置、方差以及各样本值的归属码字---------------')
            # 存储重新分组后的数据，每次循环完清空,并变成list
            data_star = [[] for ii in range(code_num)]
            # 存储处理后的三维的数据格式，每次循环完清空,并变成list
            data_star_pro = [[] for ii in range(code_num)]
            #第一小步 求第j个样本和每个码字之间的variance,并找出最小值，划分样本所属的新码字，记录新的variance
            for j in range(data.shape[0]):
                for jj in range(2**(i+1)):
                    data2[jj] = (data[j] - code_star[jj])**2 # 第j个样本和每个码字的平方和
                # print(data2)
                data_min = data2.sum(axis=1)
                # print(data_min)

                # 选出样本对应的最小方差以及相应的码字
                min_variance = data_min[0]
                min_index = 0
                for jj in range(2**(i+1)):
                    if min_variance > data_min[jj] and data_min[jj] != 0:
                        min_variance = data_min[jj]  # 第j个样本的最小方差是第jj个种子点
                        min_index = jj

                iiiii += 1
                # 注意：此时dara_star中储存的数据是二维的，到后面要处理一下
                data_star[min_index] = np.append(data_star[min_index], data[j])

            # 更新码字位置
            for jj in range(code_num):
                data_star[jj] = np.array(data_star[jj])
                if len(data_star[jj]) == 0:
                    continue
                # 把数据处理成三维的，方便使用
                for jjj in range(int(len(data_star[jj])/2)):
                    test = [data_star[jj][jjj * 2] , data_star[jj][jjj * 2 + 1]]
                    data_star_pro[jj].append(test)

                data_star_pro[jj] = np.array(data_star_pro[jj])
                code_star[jj] = data_star_pro[jj].sum(axis=0)/len(data_star_pro[jj])

            # 更新方差
            variance_star = 0
            for jj in range(2**(i+1)):
                for jjj in range(len(data_star_pro[jj])):
                    x = (data_star_pro[jj][jjj][0] - code_star[jj, 0])**2
                    y = (data_star_pro[jj][jjj][1] - code_star[jj, 1])**2
                    variance_star += (x + y)

            # 平均方差
            variance_star /= data.shape[0]
            # print(variance_star)
            # print(variance - variance_star)


            print(' 第'+str(i+1)+'次分裂的第'+str(iii)+'次更新码字位置：')
            print(code_star)
            print(' 第'+str(i+1)+'次分裂的第'+str(iii)+'次更新方差：  ' + str(variance_star))
            print(' 第'+str(i+1)+'次分裂的第'+str(iii)+'次更新各样本值归属码字：')
            for jjjj in range(code_num):
                print("     第"+ str(jjjj+1)+ "个码字有" + str(int(len(data_star_pro[jjjj]))) + "个样本值")
            iii += 1


        print(' 第'+str(i+1)+'次分裂前后减少的方差：' + str(variance_i - variance_star))
    print('*********************************VQ_LBG循环结束************************************8')

    # 输出都变成数组，好画图
    code_star = np.array(code_star)
    data_star_pro = np.array(data_star_pro, dtype=object)
    for i in range(code_num):
        data_star_pro[i] = np.array(data_star_pro[i])

    # 返回各码字位置、各码字所属样本值、方差
    return code_star, data_star_pro, variance_star

# 输入样本点(数组)，参数(结束条件)，是否已知种子点个数,(已知种子点个数，)
# 输出各个中心点的位置，各个中心点位置对应的样本
# know_cla_num = True是已知类别数, 类别数
# know_cla_num = False是未知类别数，迭代终止参数
def K_means(data, know_cla_num = False, *para):
    # 已知类别数
    # 1.从样本中选择 K 个点作为初始质心（完全随机）
    # 2.计算每个样本到各个质心的距离，将样本划分到距离最近的质心所对应的簇中
    # 3.计算每个簇内所有样本的均值，并使用该均值更新簇的质心
    # 4.重复步骤 2 与 3 ，直到达到以下条件之一：
    #       质心的位置变化小于指定的阈值（默认为 0.0001）
    #       达到最大迭代次数
    if know_cla_num:
        cla_num = int(para[0])

        # 随机生成cla_num个种子点
        # code_random = np.random.seed(123)
        code_random = np.random.seed(int(time.time()))
        code_random = np.random.randint(0 ,len(data) ,cla_num)
        code_star = np.array([data[i] for i in code_random]) # k个种子点
        code_star_before = np.array([data[i] for i in code_random]) # k个种子点
        print(code_star)

        data_star = [[] for i in range(cla_num)]

        code_star_dis = 1
        #   *************************************以下是要循环的部分****************************
        while code_star_dis != 0:
            #第j个样本与每个种子点计算距离，并更新种子点位置，更新种子点与原来种子点位置的距离，计算平均方差
            data_star = [[] for i in range(cla_num)] # 用来存放划分进第jj个种子点的样本点，每次循环要清零
            data2 = [[] for i in range(cla_num)]

            # python小知识点：矩阵变量传递的是地址，因此用FOR循环
            for i in range(cla_num):
                code_star_before[i, 0] = code_star[i, 0]
                code_star_before[i, 1] = code_star[i, 1]

            variance = 0
            # 对每个样本点进行一个划分
            for j in range(data.shape[0]):  # 20
                for jj in range(cla_num):
                    data2[jj] = (data[j] - code_star[jj])**2 # 第j个样本和每个码字的平方和
                # print(data2)
                data2 = np.array(data2)
                data_min = data2.sum(axis=1)
                # print(data_min)

                # 选出样本对应的最小方差以及相应的码字
                min_variance = data_min[0]
                min_index = 0
                for jj in range(cla_num):
                    if min_variance > data_min[jj]:
                        min_variance = data_min[jj] #第j个样本与第jj个种子点的方差最小
                        min_index = jj
                # print(min_variance, min_index)
                # 将第j个样本划分进第jj个种子点的范围
                data_star[min_index].append(data[j].tolist())


            # 更新第j个中心点的位置
            for j in range(cla_num):
                x ,y =0, 0
                if len(data_star[j]) == 0:
                    continue
                for jj in range(len(data_star[j])):
                    x += data_star[j][jj][0]
                    y += data_star[j][jj][1]
                code_star[j, 0] = x/len(data_star[j])
                code_star[j, 1] = y/len(data_star[j])

            # 更新方差************************************
            variance = 0
            for i in range(cla_num):
                for j in range(len(data_star[i])):
                    x = (data_star[i][j][0] - code_star[i, 0])**2
                    y = (data_star[i][j][1] - code_star[i, 1])**2
                    variance += (x + y)
            # 平均方差
            variance /= data.shape[0]
            print("方差变化")
            print(variance)
            # print(data_star)  # list

            # 看看中心点位置移动了多少
            code_star_dis = ((code_star - code_star_before)**2).sum()

            # print(code_star_dis)
        print("循环结束")
        print(code_star)

        data_star = np.array(data_star, dtype=object)  # 没有这个dtype就会报错，好像是因为列表不整齐，直接转数组会报错
        for i in range(cla_num):
            data_star[i] = np.array(data_star[i])
        return code_star, data_star

    # 不知类别数
    # 1.计算与每个样本点方差,确定方差最小的种子点位置
    # 2.增加一个种子，利用方差将样本点划分进最近的种子点，利用新的样本点更新方差最小的种子点位置
    # 3.判断当前j个种子点总方差与j-1个种子点总方差大小，若减小的比例大于0.1，则重复第2步
    # 4.选出第j-1个种子点的位置，及每个种子点拥有的样本点
    else:
        para = para[0]

        # 计算第一个起始种子点的位置
        x, y = 0, 0
        for i in range(data.shape[0]):
            x += data[i, 0]
            y += data[i, 1]
        code_star = [[x/data.shape[0], y/data.shape[0]]]

        # 计算只有一个种子点的方差
        variance_befor = 0
        for j in range(data.shape[0]):
            temp = (data[j] - code_star[0])**2
            temp = np.array(temp)# 改成数组好用sum()函数
            variance_befor += temp.sum()
        variance_befor /= data.shape[0]

        variance = 0
        # 只有一个种子点的
        data_star = []
        data_star.append(data.tolist()) # 就直接全部存到data_star里 方便第一次while循环时给data_stra_before赋值
        # print(data_star)
        code_star_before = [[0, 0] for i in range(len(code_star))]  # 存放上一循环的种子点
        data_star_before = [[] for i in range(len(code_star))]  # 存放上一循环的样本点
        # **************************************while大循环*********************************
        if (variance_befor - variance)  / variance_befor < para:
            print("请减小参数的值后，重新运行代码！")
            return
        while (variance_befor - variance)  / variance_befor > para :

            # nonlocal code_star_before  #  nonlocal关键字，用来在函数或其他作用域中使用外层(非全局)变量。
            code_star_before = [[0, 0] for i in range(len(code_star))]  # 存放上一循环的种子点
            data_star_before = [[] for i in range(len(code_star))]  # 存放上一循环的样本点
            # 储存上一次j-1个种子点的样本点分布
            for i in range(len(code_star)): #0,1,2
                for j in range(len(data_star[i])):
                    data_star_before[i].append(data_star[i][j])

            # 将前一次的方差储存到新的地方
            # print(len(code_star))
            if len(code_star) != 1:
                variance_befor = variance
                variance = 0
            # 将前一次循环种子点位置储存到新的地方
            for i in range(len(code_star)):
                code_star_before[i][0] = code_star[i][0]
                code_star_before[i][1] = code_star[i][1]

            # 加一个种子点
            print("加一个种子点")
            print(data[np.random.randint(0 ,len(data))])
            code_star.append(data[np.random.randint(0 ,len(data))])
            code_star_dis = 1  # 随便设置一个值，只要能进入while小循环就好
            # 从这里加一个while小循环，直到每个种子点位置变化小于para，则聚类完毕，再计算平均方差，最后进入第一个while与上次的平均方差比较，决定是否再增加种子点
            while code_star_dis != 0 :
                data2 = [[] for i in range(len(code_star))]
                data_star = [[] for i in range(len(code_star))]  #每次循环都要清空才能记录本次循环的样本点

                # 将上一次while小循环的种子点位置记录下来,方便每次小循环计算种子点位置变化
                code_star_temp = [[0 ,0] for i in range(len(code_star))] # 直接用列表的append还是赋值地址，只能是每个元素进行赋值
                for i in range(len(code_star)):
                    code_star_temp[i][0] = code_star[i][0]
                    code_star_temp[i][1] = code_star[i][1]

                # 利用距离划分样本点
                for j in range(data.shape[0]):  # 20
                    for jj in range(len(code_star)):
                        data2[jj] = (data[j] - code_star[jj])**2 # 第j个样本和每个码字的平方和

                    data2 = np.array(data2)
                    data_min = data2.sum(axis=1).tolist()
                    # print(data_min)

                    # 选出样本对应的最小方差以及相应的码字
                    # min_variance = min(data_min)
                    min_index = data_min.index(min(data_min))

                    # 将第j个样本划分进第jj个种子点的范围
                    data_star[min_index].append(data[j].tolist())

                # 更新种子点的位置
                for j in range(len(code_star)):
                    x ,y =0, 0
                    if len(data_star[j]) == 0:
                        continue
                    for jj in range(len(data_star[j])):
                        x += data_star[j][jj][0]
                        y += data_star[j][jj][1]

                    code_star[j][0] = x/len(data_star[j])
                    code_star[j][1] = y/len(data_star[j])

                # 更新方差信息variance
                variance = 0
                for i in range(len(code_star)):
                    for j in range(len(data_star[i])):
                        # code_star[i][0] - data_star[i][j][0]
                        x = (code_star[i][0] - data_star[i][j][0])**2
                        y = (code_star[i][1] - data_star[i][j][1])**2
                        variance += x
                        variance += y
                variance /= data.shape[0] # 平均方差

                # print(str(len(code_star)) + " 个种子点的方差更新：" + str(variance))


                # 计算种子点变化距离
                x ,y = 0 ,0
                code_star_dis = 0
                for i in range(len(code_star)):
                    x = (code_star[i][0] - code_star_temp[i][0])**2
                    y = (code_star[i][1] - code_star_temp[i][1])**2
                    code_star_dis += x + y

            print("有 " + str(len(code_star)) + " 个种子点的最小方差：" + str(variance))
            print("有 " + str(len(code_star) - 1) + " 个种子点的最小方差：" + str(variance_befor))
            print((variance_befor - variance)  / variance_befor,para)

        print("**********************************************************************")
        print("有效种子点")
        print(code_star_before)
        print("平均方差")
        print(variance_befor)
        print("样本点划分")
        for i in range(len(code_star_before)):
            print(len(data_star_before[i]))
        # 输出各个中心点的位置，各个中心点位置对应的样本

        for i in range(len(data_star_before)):
            data_star_before[i] = np.array(data_star_before[i])
        data_star_before = np.array(data_star_before, dtype=object)
        code_star_before = np.array(code_star_before)
        return code_star_before,data_star_before

def main():
    # ****************************************画图预处理************************************************
    plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


    # ****************************************生 成 数 据************************************************
    # 样本点总数量
    sample_num = 200
    # 生成数据的中心点位置
    cla_locat = [[0,0],[10,0],[0, 10],[10, 10]]

    # 生成数据(dim=3)
    data = Databuild(sample_num, len(cla_locat), cla_locat)

    # 画图
    for i in range(data.shape[0]):
        plt.scatter(data[i,:,0],data[i,:,1])
    plt.title('模式生成函数生成样本点', fontsize = 22)
    plt.xlabel('x坐标', fontsize = 16)
    plt.ylabel('y坐标', fontsize = 16)
    plt.show()

    # 降维
    data1 = list()
    for i in range(data.shape[0]):
        data1.extend(data[i])
    data1 = np.array(data1)


    # # ****************************************矢 量 量 化************************************************
    # # 参数  小一点
    # para1 = 0.001
    # # 希望生成的码字个数
    # code_num = 4
    # # 输入码字个数、参数、生成的数据
    # # 返回各码字位置(list格式)、各码字所有的样本值(list格式,dim=3)、方差
    # a = time.time()
    # code_star_LBG, data_LBG, variance_LBG = VQ_LBG(code_num, para1 ,data1)
    # b = time.time()
    # print("聚类所用时间")
    # print(b-a)
    # print("VQ_LBG各个码字最终位置：")
    # print(code_star_LBG)
    # print("最终方差：")
    # print(variance_LBG)
    #
    #
    # # print(data_LBG[0][0 : (data_LBG.shape[0]-1)][0])
    # # VQ_LBG画图
    # for i in range(code_num):
    #     if data_LBG[i].shape[0] == 0:
    #         continue
    #     plt.scatter(data_LBG[i][: ,0] , data_LBG[i][: ,1]) # 列表切片
    # # print(code_star_LBG[0, 0])
    # plt.scatter(code_star_LBG[:, 0], code_star_LBG[:, 1], s=100,  color='k', marker= '*')
    # for i in range(code_star_LBG.shape[0]):
    #     plt.text(round(code_star_LBG[i, 0], 2), round(code_star_LBG[i, 1], 2),
    #              (round(code_star_LBG[i, 0], 2), round(code_star_LBG[i, 1], 2)),ha='center', va='bottom', fontsize=17)
    # plt.title('VQ-LBG算法k = '+str(code_num)+'聚类结果', fontsize = 22)
    # plt.xlabel('x坐标', fontsize = 16)
    # plt.ylabel('y坐标', fontsize = 16)
    # plt.show()


    # ********************************************K—means************************************************
    # **************************已知类别数K—means*********************************
    # # 输入样本点(数组)，参数(结束条件)，是否已知
    # # 输出各个中心点的位置，各个中心点位置对应的样本
    # # know_cla_num = True是已知类别数, 类别数
    # para = 4  #  类别数
    # a = time.time()
    # code_star_Kmeans ,data_Kmeans = K_means(data1, True, para)
    # b = time.time()
    # print("时间")
    # print(b-a)
    #
    # # K_means画图
    # for i in range(para):
    #     plt.scatter(data_Kmeans[i][:, 0], data_Kmeans[i][:, 1])
    # plt.scatter(code_star_Kmeans[:, 0], code_star_Kmeans[:, 1], s=100,  color='k', marker= '*')
    # # for a, b in zip(code_star_Kmeans[0], code_star_Kmeans[1]):
    # for i in range(code_star_Kmeans.shape[0]):
    #     plt.text(round(code_star_Kmeans[i, 0], 2), round(code_star_Kmeans[i, 1], 2),
    #              (round(code_star_Kmeans[i, 0], 2), round(code_star_Kmeans[i, 1], 2)),ha='center', va='bottom', fontsize=17)
    # plt.title('已知类别数K均值算法k = ' + str(para) + "聚类结果", fontsize = 22)
    # plt.xlabel('x坐标', fontsize = 16)
    # plt.ylabel('y坐标', fontsize = 16)
    # # plt.savefig('./已知类别数k='+str(para)+'.png')
    # plt.show()
    # for i in range(para):
    #     print(len(data_Kmeans[i]))

    # **************************未知类别数K—means*********************************

    # know_cla_num = False是未知类别数，迭代终止参数(参数越小，生成的种子点数越多) 0.5什么的可能会比较好
    #0.38-0.1
    para = 0.3
    a = time.time()
    code_star_Kmeans ,data_Kmeans = K_means(data1, False, para)
    b = time.time()
    print(b-a)
    # print(data_Kmeans.shape)
    print("划分成" + str(len(code_star_Kmeans)) + "个样本点")
    for i in range(data_Kmeans.shape[0]):
        plt.scatter(data_Kmeans[i][:, 0], data_Kmeans[i][: ,1])
    plt.scatter(code_star_Kmeans[:, 0], code_star_Kmeans[:, 1], s=100,  color='k', marker= '*')
    for i in range(code_star_Kmeans.shape[0]):
        plt.text(round(code_star_Kmeans[i, 0], 2), round(code_star_Kmeans[i, 1], 2),
                 (round(code_star_Kmeans[i, 0], 2), round(code_star_Kmeans[i, 1], 2)),ha='center', va='bottom', fontsize=17)
    plt.title('未知类别数K均值算法k = 4聚类结果' , fontsize = 22)
    plt.xlabel('x坐标', fontsize = 16)
    plt.ylabel('y坐标', fontsize = 16)
    plt.show()

if __name__ == '__main__':
    main()


