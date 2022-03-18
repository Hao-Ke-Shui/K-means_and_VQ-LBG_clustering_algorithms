# K-means_and_VQ-LBG_clustering_algorithms
# numpy实现K均值算法(可选是否已知类别数)和VQ-LBG聚类算法
只有一个main文件，里面有四个函数

1、数据生成函数
  def Databuild(sample_num, cla_num, *cla_locat)  
  # 输入依次是样本数, 类别数，类别中心
  # 输出数据数组

2、VQ-LBG函数
  def VQ_LBG(code_num, para1 ,data)
  # 输入依次是码字个数、参数、数据
  # 输出依次是各码字位置、各码字所含样本点
  
3、K-menas函数
  def K_means(data, know_cla_num = False, *para)
  # 当 know_cla_num=True 时，为已知类别数的K-means
    # 输入依次是数据、True、类别数
    # 输出依次是各聚类中心位置、各聚类中心所含样本点
  # 当 know_cla_num=False 时，为未知类别数的K-means
    # 输入依次是数据、False、迭代终止参数
    # 输出依次是各聚类中心位置、各聚类中心所含样本点
 
4、主函数
  def main()
  # 用来生成数据、进行聚类操作、画图

  
