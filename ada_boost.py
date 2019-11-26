import math
import pandas as pd
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('D:\可爱臭鸭鸭\duck_data.csv')#需要添加目标文件目录
train_data = np.array(df.iloc[:,1:10])
attributeMap={}
attributeMap['white']=1
attributeMap['yellow']=0.5
attributeMap['black']=0
#====================
attributeMap['curl']=1
attributeMap['common']=0.5
attributeMap['straight']=0
#====================
attributeMap['dull']=0
attributeMap['loud']=0.5
attributeMap['clear']=1
#====================
attributeMap['hard']=0
attributeMap['common']=0.5
attributeMap['soft']=1
#====================
attributeMap['small']=0
attributeMap['common']=0.5
attributeMap['big']=1
attributeMap['ugly']=0
attributeMap['cute']=1
attributeMap['no']=-1
attributeMap['yes']=1
#============数据化=================
for i in range(len(train_data)):
  for j in range(len(train_data[0])):
      if j != 6 and j != 7:
        train_data[i,j]=attributeMap[train_data[i,j]]
#============归一化=================
food=train_data[:,6]
health=train_data[:,7]
food_min=np.min(food)
food_max=np.max(food)
health_min=np.min(health)
health_max=np.max(health)
for i in range(len(food)):
    food[i]=( food[i]-food_min)/(food_max-food_min)
    food[i]=round( food[i],2)
for i in range(len(health)):
    health[i] = (health[i]- health_min) / (health_max - health_min)
    health[i] = round(health[i],2)

row_rand_array = np.arange(train_data.shape[0])
np.random.shuffle(row_rand_array)
row_rand = train_data[row_rand_array[0:10]]#抽取12条数据训练
rest_rand = train_data[row_rand_array[10:17]]#抽取12条数据训练
train_data=row_rand
test_data=rest_rand
test_label=test_data[:,8]
train_label=train_data[:,8]
train_data=np.delete(train_data,8,1)#第三个数1表示列，0表示行
test_data=np.delete(test_data,8,1)#第三个数1表示列，0表示行
m,n=np.shape(train_data)
c,d=np.shape(test_data)

#得到初始数据权重D1===============
def get_initial_D(train_data):
  T=20#学习器上限
  D=[]
  for i in range(len(train_data)):
   n=1/len(train_data)
   D.append(n)
  iter=0
  return D
#得到第i个属性的最佳分类器（弱分类器）===============
def get_weak_classifier(train_data,a,D):
    data=train_data[:,a]
    value=[]
    for i in data:
        if i not in value:
            value.append(i)
    value=np.sort(value)#排序=====
    classify_point=[]
    for i in range(len(value)-1):
        mid=(value[i]+value[i+1])/2
        classify_point.append(mid)
    classifer={}#分类器集合==
    for point in classify_point:
        predict_data=[]
        for i in range(len(data)):
            if data[i]>point:
                value=1
            else:
                value =-1
            predict_data.append(value)
        e1=0
        for i in range(len(train_label)):
            if predict_data[i]!=train_label[i]:
                e1+=D[i]
        for i in range(len(data)):
            if data[i]<point:
                value=1
            else:
                value =-1
            predict_data.append(value)
        e2=0
        for i in range(len(train_label)):
            if predict_data[i]!=train_label[i]:
                e2+=D[i]
        classifer[point]=[]
        classifer[point].append(e1)
        classifer[point].append(e2)
    min_classifier=classify_point[0]
    min_e=classifer[min_classifier][0]
    direction=0
    for point in classifer.keys():
        for i in range(len(classifer[point])):
            if classifer[point][i]<min_e:
                min_e=classifer[point][i]
                min_classifier=point
                direction=i
    if direction==0:
        direction='forward'
    else:
        direction='back'
    return min_e,min_classifier,direction

def predict_data_using_G(train_data,direction,classifier,a):
    data=train_data[:,a]
    predict=[]
    for i in range(len(data)):
        if direction=='forward':
            if data[i]>classifier:
                pre=1
            else:
                pre=-1
        else:
            if data[i]<classifier:
                pre=1
            else:
                pre=-1
        predict.append(pre)
    return predict




'''
D=get_initial_D(train_data)
min_e,min_classifier,direction=get_weak_classifier(train_data,0,D)
print(min_e,min_classifier,direction)
'''
iter=0
T=n#最大学习器数目
D=get_initial_D(train_data)#弱学习器初始化
a_list=[]#最终的基学习器权重列表
G_list=[]
f=[]#弱学习器的线性组合
for i in range(len(D)):
    f.append(0)
while iter<T :
 e,classifier,direction=get_weak_classifier(train_data,iter,D)
 try:
   a=1/2*np.log((1-e)/e)   #弱学习器G(iter)的系数
 except ZeroDivisionError:
   a=10
 G_predict=predict_data_using_G(train_data,direction,classifier,iter)
 Z=0
 for i in range(len(train_data)):
    Z+=D[i]*np.exp(-a*train_label[i]*G_predict[i])
 D_new=[]
 for i in range(len(D)):
    new=D[i]/Z*np.exp(-a*train_label[i]*G_predict[i])
    D_new.append(new)
 D=D_new#数据权重更新完成
 for i in range(len(f)):
     f[i]=f[i]+G_predict[i]*a
 G={'classifier_point':classifier,'direction':direction}
 G_list.append(G)
 a_list.append(a)
 iter+=1

#=====================================
def get_train_data_accuracy(f):
  right=0
  for i in range(len(f)):
    if f[i]>0:
        f[i]=1
    else:
        f[i]=-1
    if f[i]==train_label[i]:
        right+=1
  right=right/len(train_label)*100
  print ('正确率为：'+str(right)+'%')

result=[]
for i in range(len(test_label)):
    result.append(0)
for j in range(len(G_list)):
   pre_y=predict_data_using_G(test_data,G_list[j]['direction'],G_list[j]['classifier_point'],j)
   for i in range(len(result)):
       result[i]+=pre_y[i]*a_list[j]
right=0
for i in range(len(result)):
    if result[i]>0:
        result[i]=1
    else:
        result[i]=-1
    if result[i] == test_label[i]:
        right += 1

right=right/len(test_label)*100
print ('正确率为：'+str(right)+'%')


