
#运行本notebook需要以下库，如本cell报错，请先安装库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
from sklearn.model_selection import KFold
from scipy.io import loadmat

def matimport(path): #返回训练值和标记值
    m = loadmat(path)
    return m['feature'],m['label']


#使用其他小组的数据做测试
source1,source2=matimport("C:\\Users\\jhong\\OneDrive\\当前项目\\ML3-RBF-Regression\\其他资料\\测试数据(所有人)\\5\\train.mat")    #特征值
ip_x2=np.array(source1)                       #特征值矩阵
ip_y2=np.array(source2)                      #label矩阵
ip_y2=ip_y2.reshape(720,1)               #转置便于后面的计算

#自己小组的数据用来训练模型
sourcedata_feature,sourcedata_label=matimport("C:\\Users\\jhong\\OneDrive\\当前项目\\ML3-RBF-Regression\\测试数据(小组)\\train.mat")
total_data_number=sourcedata_label.shape[1]   #总的样本数据个数
ip_x=np.array(sourcedata_feature)                  #特征值矩阵
ip_y=np.array(sourcedata_label)                     #label矩阵
ip_y=ip_y.reshape(720,1)                        #转置便于后面的计算
train_number=int(total_data_number*10/10);      #用来训练模型的样本个数,由于用了其他小组的数据作为测试集故可用自己的全部的数据训练模型

def MED(point1,point2):         #计算两点之间的欧式距离
    return (np.sum((point1-point2)**2))**0.5
    
def Gaussian(point1,point2,variance):  #径向基函数、高斯函数
    return math.exp(-(MED(point1,point2))**2/(2*variance**2))

def Output_Matrix(input_y):           #对label值进行一些调整，把0变为[1,0,0],1变[0,1,0],2变[0,0,1]
    size=input_y.shape[0]           
    op_y=np.zeros((size,3))
    for i in range(size):
        if input_y[i]==0:
            op_y[i][0]=1
        elif input_y[i]==1:
            op_y[i][1]=1
        else:
            op_y[i][2]=1
    return op_y

def get_variance(center_point_number,center_point):   #计算方差
    max_MED=0
    for i in range(center_point_number):
        for j in range(center_point_number):
            med_=MED(center_point[i],center_point[j])
            if med_>max_MED:
                max_MED=med_
    return max_MED/((2*center_point_number)**0.5)

def get_wight(ip_x,center_point,center_point_number,op_y,variance):          #计算权值矩阵
    weight=np.zeros((center_point_number,3))      #权重矩阵
    j_matrix=np.zeros((center_point_number,train_number)) #激励矩阵
    for i in range(center_point_number):
        for j in range(train_number):
            j_matrix[i][j]=Gaussian(center_point[i],ip_x[j], variance)
    j_inverse=np.linalg.pinv(j_matrix)    #激励矩阵的伪逆
    weight=np.dot(j_inverse.T,op_y)
    return weight 
    

def tcal_y(input_x,weight):        #由神经元的输出乘以权重计算最终输出
    size=input_x.shape[0]
    y=np.array((size,3))
    y=np.dot(input_x,weight)
    return y


def cal_hide_output_matrix(input_x,center,center_point_number,variance):   #计算所有神经元的输出
    size=input_x.shape[0]                      #神经元个数
    hide_output_matrix=np.zeros((size,center_point_number))    
    for i in range(size):
        for j in range(center_point_number):   
            hide_output_matrix[i][j]=Gaussian(input_x[i],center[j], variance)  #利用高斯函数进行映射
    return hide_output_matrix

def calc_center_point(cluster):         #计算簇的中心点
    size_of_cluster = len(cluster)            #簇中的点的个数
    center_point=np.zeros((1,70))
    for i in range(size_of_cluster):
        center_point=center_point+cluster[i]         #所有点加起来再除以个数
    center_point = center_point*(1/size_of_cluster)
    return center_point
 
    
def point_no_equal(point1,point2):   #判断两个中心点是否重合
    size=point1.shape[0]
    for i in range(size):
        if point1[i]!=point2[i]:
            return True
    return False
 

def check_center_diff(center, new_center):    #检查迭代之后形成的所有中心点与上一次是否相同
    n = len(center)
    for i in range(n):
        if point_no_equal(new_center[i],center[i]):
            return False
    return True




def k_means(input_x,k):  #k—means聚类方法
    center=input_x[:k]     #先曲前k个点作为中心点
    size=input_x.shape[0]   #总的点数
    cluster=[]              
    max_n=0              #最大迭代次数
    while True:
        for i in range(k):
            cluster.append([])
        for i in range(size):    #遍历每个点找到属于哪个簇
            min=1e5
            min_index=-1
            for j in range(k): 
                med_=MED(input_x[i],center[j])
                if min>med_:
                    min=med_
                    min_index=j
            cluster[min_index].append(input_x[i])
        temp_center=np.zeros((k,input_x.shape[1]))   #新的中心点
        for i in range(k):
            temp_center[i]=calc_center_point(cluster[i])
        if check_center_diff(center, temp_center):   #检查新旧的各个中心点是否相同，若相同停止聚类
            break
        max_n+=1
        if max_n>100:
            break
        center=temp_center    #更新所有中心点
        cluster=[]
    return center


center_point_number=720         #隐藏神经元的数量，可改变来改变隐藏层的规模
k_center=k_means(ip_x[:train_number],center_point_number)  #k-means聚类
        

var=get_variance(center_point_number,k_center)    #高斯函数的方差
output=Output_Matrix(ip_y)                        #改变训练集的输出
output2=Output_Matrix(ip_y2)                      #改变测试集的输出
weight=get_wight(ip_x[:train_number],k_center,center_point_number,output[:train_number] , var) #计算权重
hide_output_matrix=cal_hide_output_matrix(ip_x2[:],k_center, center_point_number,var)  #计算神经元输出
y=tcal_y(hide_output_matrix,weight)          #计算最总输出
accurate=0                        #正确的个数
for i in range(total_data_number):   #将输出的三个值先进行比较最大的变为1，剩下的全为0，与output统一
    if y[i][0]>y[i][1] and y[i][0]>y[i][2]:
        y[i][0]=1
        y[i][1]=0
        y[i][2]=0
    elif y[i][1]>y[i][0] and y[i][1]>y[i][2]:
        y[i][0]=0
        y[i][1]=1
        y[i][2]=0
    else:
        y[i][0]=0
        y[i][1]=0
        y[i][2]=1
for i in range(total_data_number):    #比较相同的个数
    if y[i][0]==1 and y[i][0]==output2[i][0]:
        accurate+=1
    elif y[i][1]==1 and y[i][1]==output2[i][1]:
        accurate+=1
    elif y[i][2]==1 and y[i][2]==output2[i][2]:
        accurate+=1
print(accurate/(total_data_number))   #正确率






    
