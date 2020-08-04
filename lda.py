"""
Created on Fri Aug 30 10:06:40 2019

@author: utkarsh
"""
import imageio
import numpy as np
import  matplotlib.pyplot as plt

import os
def loadimages1(path="."):
    return[os.path.join(path,f)for f in os.listdir(path) if f.endswith('happy.gif')] 
def loadimages2(path="."):
    return[os.path.join(path,f)for f in os.listdir(path) if f.endswith('sad.gif')] 
    
#load only happy train images
train_happy=loadimages1(".data/train")
#load only sad train images
train_sad=loadimages2(".data/train")
#load only happy test images
test_happy=loadimages1(".data/test")
#load only sad test images
test_sad=loadimages2(".data/test")
train_complete=[]

def concat_test(images):
    t=[]
    for file in images:
        t.append(imageio.mimread(file))
    return t
def concat(images):
    for file in images:
        train_complete.append(imageio.mimread(file))
def reduce_extra_dimmension(img):
    a=0
    k=[]
    for i in img:
        k.append(img[a][-1])
        a=a+1
    return k
def flat(k):
    x=[]
    a=0
    for i in k:
        x.append(k[a].flatten())
        a=a+1
    return x
def normalize_mean(img,mean):
    normal=np.zeros((img.shape))
    a,b=img.shape
    for i in range(b):
        normal[:,i]=img[:,i]-mean
    return normal
def high_dimmensional_cov(normal):
    a,b=normal.shape
    "20 is number of data points"
    checkmatrix=np.matmul(np.transpose(normal),normal)/b
    return checkmatrix
def cov(normal):
    a,b=normal.shape
    "20 is number of data points"
    checkmatrix=np.matmul(normal,np.transpose(normal))/b
    return checkmatrix
def eigen_value_vector(matrix):
    return np.linalg.eig(matrix)
def final_eigen_vector(a,b,normal):
    u=np.zeros([10201,20])
    for i in range(20):
        if(a[i]!=0):
            u[:,i]=np.matmul(normal,b[:,i])/np.sqrt(a[i]*20)
    return (a,u)
def eigen_sort(value,vector):
    idx = value.argsort()[::-1]   
    eigenValues = value[idx]
    eigenVectors = vector[:,idx]
    return (eigenValues,eigenVectors)
def eig_faces(a):
    a=a.reshape(101,101)
    return a
def projection(data,v):
    return(np.matmul(data,v))
def final_projection(eigen_matrix,x,k):
    u=eigen_matrix[:,:k]
    y=np.matmul(x,u)
    return y 
concat(train_happy)
concat(train_sad)
train_complete=reduce_extra_dimmension(train_complete)
data_list=flat(train_complete)
data_matrix=np.array(data_list)
mean_data=np.mean(data_matrix,axis=0)
data_matrix_transposed=np.transpose(data_matrix)
normalized_data_matrix=normalize_mean(data_matrix_transposed,mean_data)
high_covarinace=high_dimmensional_cov(normalized_data_matrix)
eig_val,eig_vector=eigen_value_vector(high_covarinace)
eig_val=np.round((np.absolute(eig_val)))
eig_valf,eig_vectorf=final_eigen_vector(eig_val,eig_vector,normalized_data_matrix)
eig_vals,eig_vectors=eigen_sort(eig_valf,eig_vectorf)

# K=12 (PCA)
K=11
x=final_projection(eig_vectors,data_matrix,K)

# LDA Function (Takes Data Matrix as Input)

def lda(x):
    happy=x[:9,:]
    sad=x[9:20,:]
    mean_happy=np.mean(happy,axis=0)
    mean_sad= np.mean(sad,axis=0)
    happy_transpose=np.transpose(happy)
    sad_transpose=np.transpose(sad)
    normalized_happy=normalize_mean(happy_transpose,mean_happy)
    normalized_sad=normalize_mean(sad_transpose,mean_sad)
    s1=cov(normalized_happy)
    s2=cov(normalized_sad)
    sw=s1+s2
    inv=np.linalg.inv(sw)
    diff_mean=mean_happy-mean_sad
    w=np.matmul(inv,diff_mean)
    y=np.matmul(x,w)
    return y,w    
y,w=lda(x)

"testing Happy images"
test_happy=concat_test(test_happy)
data_test_happy=reduce_extra_dimmension(test_happy)
data_list=flat(data_test_happy)
data_matrix_happy_test=np.array(data_list)
t=final_projection(eig_vectors,data_matrix_happy_test,K)
y_happy=np.matmul(t,w)
"testing sad images"
test_sad=concat_test(test_sad)
data_test_sad=reduce_extra_dimmension(test_sad)
data_list_sad=flat(data_test_sad)
data_matrix_sad_test=np.array(data_list_sad)
t1=final_projection(eig_vectors,data_matrix_sad_test,K)
y_sad=np.matmul(t1,w)
#!h=y_sad.tolist()+y_happy.tolist()
h=np.array(y)
m=np.mean(h)
plt.figure()
plt.subplot(211)
plt.plot(y[0:9],[1,1,1,1,1,1,1,1,1],'go',label='Happy_Train_Data(9 images)',markersize=12)
plt.plot(y[9:20],[1,1,1,1,1,1,1,1,1,1,1,],'yh',marker='+',ms=14,label='Sad_Train_Data(11 images)',markersize=20)
plt.plot(y_happy[0:5],[1,1,1,1,1],'mo',label='Happy_Test_Data(6 images)')
plt.plot(y_sad[0:4],[1,1,1,1],"c^",marker='+',ms=14,label='Sad_Test_Data(4 images)')
plt.axvline(x=m,label='Threashold')
plt.xlabel("1 Dimmensional LDA Projection axis",fontsize='medium',c='c')
plt.title("K=12(For PCA) Gives Maximum Seperablity")
plt.legend(loc='upper left')
plt.show()
