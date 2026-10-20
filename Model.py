import numpy as np
import matplotlib.pyplot as plt
import copy
from data import x_data,y_train
def z_normalize(x):
    mu=np.mean(x,axis=0)
    sigma=np.std(x,axis=0)
    x_norm=(x-mu)/sigma
    return x_norm, mu, sigma
def compute_cost(x,y,w,b):
    m,n=x.shape
    fwb=np.dot(x,w)+b
    diff=fwb-y
    cost=(np.sum(diff**2))/(2*m)
    return cost
def compute_gradient(x,y,w,b):
    m,n=x.shape
    fwb=np.dot(x,w)+b
    diff=fwb-y
    dj_dw=np.dot(diff,x)/m
    dj_db=np.sum(diff)/m
    return dj_dw, dj_db
def gradient_descent(x,y,w_in,b_in,alpha,num_itr,cost,gradient):
    w=copy.deepcopy(w_in)
    b=b_in
    J=0.0
    Jv=[]
    print(f"Cost value: {J}")
    print(f"Parameter values(w,b): {w},{b}")
    print("\n=========================================\n")
    for i in range(num_itr):
        dj_dw, dj_db = gradient(x,y,w,b)
        w-=(alpha*dj_dw)
        b-=(alpha*dj_db)
        J=cost(x,y,w,b)
        Jv.append(J)
        if (i+1)%1000==0:
            print(f"Iteration count: {i+1}")
            print(f"Cost value: {J}")
            print(f"Parameter values(w,b): {w},{b}")
            print("\n=========================================\n")
    return w,b,J,Jv
xsq=x_data**2
x1=(x_data[:,0]*x_data[:,5]).reshape(-1, 1)
x2=(x_data[:,4]*x_data[:,5]).reshape(-1, 1)
x3=(x_data[:,2]*x_data[:,3]).reshape(-1, 1)
x4=(x_data[:,3]*x_data[:,5]).reshape(-1, 1)
x5=(x_data[:,0]*x_data[:,3]).reshape(-1, 1)
x_data=np.column_stack([x_data,xsq,x1,x2,x3,x4,x5])
x_train, mu, sigma=z_normalize(x_data)
m,n=x_train.shape
wf,bf,J,Jv=gradient_descent(x_train,y_train,np.zeros(n,),0.0,0.003,12000,compute_cost,compute_gradient)
fwb=np.dot(x_train,wf)+bf
fwb=np.round(fwb).astype(int)
r2 = 1 - (np.sum((fwb - y_train)**2) / np.sum((y_train - np.mean(y_train))**2))
print(f"Predicted values: {fwb[:5]}")
print(f"Actual values: {y_train[:5]}")
r2 = 1 - (np.sum((fwb - y_train)**2) / np.sum((y_train - y_train.mean())**2))
print(f"R²: {r2}")
fig,ax = plt.subplots(2,3)
ax[0,0].scatter(x_data[:,0],y_train,color='b',marker='.',s=3)
ax[0,0].set_xlabel('CPU number of cores')
ax[0,0].set_ylabel('FPS')
ax[0,1].scatter(x_data[:,1],y_train,color='b',marker='.',s=3)
ax[0,1].set_xlabel('CPU number of threads')
ax[0,1].set_ylabel('FPS')
ax[0,2].scatter(x_data[:,2],y_train,color='b',marker='.',s=3)
ax[0,2].set_xlabel('CPU turbo clock')
ax[0,2].set_ylabel('FPS')
ax[1,0].scatter(x_data[:,3],y_train,color='b',marker='.',s=3)
ax[1,0].set_xlabel('GPU boost clock')
ax[1,0].set_ylabel('FPS')
ax[1,1].scatter(x_data[:,4],y_train,color='b',marker='.',s=3)
ax[1,1].set_xlabel('GPU vram in gb')
ax[1,1].set_ylabel('FPS')
ax[1,2].scatter(x_data[:,5],y_train,color='b',marker='.',s=3)
ax[1,2].set_xlabel('GPU number of CUDA/shading units')
ax[1,2].set_ylabel('FPS')
fig, ax1=plt.subplots()
ax1.plot(Jv,color='r')
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Cost values")
ax1.set_title("Cost vs iterations")
plt.show()