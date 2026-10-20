import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
W=[]
B=[]
def compute_cost(x,y,w,b):
    m,n=x.shape
    fwb=np.dot(x,w)+b
    diff=fwb-y
    cost=(np.sum(diff**2))/(2*m)
    return cost
def compute_gradient(x,y,w,b):
    m,n=x.shape
    fwb=w*x+b
    diff=fwb-y
    dj_dw=np.sum(diff*x)/m
    dj_db=np.sum(diff)/m
    return dj_dw, dj_db
def gradient_descent(x,y,w,b,a,itr):
    Jv=[]
    J=0.0
    for i in range(itr):
        dj_dw,dj_db=compute_gradient(x,y,w,b)
        J=compute_cost(x,y,w,b)
        W.append(w)
        B.append(b)
        Jv.append(J)
        w-=a*dj_dw
        b-=a*dj_db
    return w,b,J,Jv
def run_visualizer(w,b,a,itr):
    W.clear()
    B.clear()
    x_data=np.random.rand(100).reshape(-1,1)
    p=random.uniform(-5.0,5.0)
    q=random.uniform(-3.0,3.0)
    y_data=p*x_data+q
    wf,bf,J,Jv=gradient_descent(x_data,y_data,w,b,a,itr)
    W_m,B_m=np.meshgrid(np.array(W),np.array(B))
    m,n=x_data.shape
    fwb=x_data*W_m.ravel()[None,:]+B_m.ravel()[None,:]
    diff=fwb-y_data
    cost=np.sum(diff**2,axis=0)/(2*m)
    cost=cost.reshape(W_m.shape)
    z=compute_cost(x_data,y_data,p,q)
    fig,ax=plt.subplots(figsize=(10,6),subplot_kw={'projection':'3d'})
    ax.plot_surface(W_m, B_m, cost, cmap='viridis', alpha=0.5)
    line,=ax.plot([],[],[],color='red')
    point,=ax.plot([W[0]],[B[0]],[Jv[0]],color='red',markersize=10,linestyle='')
    info=ax.text2D(0.05,0.95,'',transform=ax.transAxes)
    def update(i):
        line.set_data(W[:i+1],B[:i+1])
        line.set_3d_properties(Jv[:i+1])
        point.set_data([W[i]],[B[i]])
        point.set_3d_properties([Jv[i]])
        info.set_text(f"Iteration: {i+1}\nw={W[i]:.4f}\nb={B[i]:.4f}\nCost={Jv[i]:.6f}")
        return line,point,info
    ani = FuncAnimation(fig,update,frames=range(0,len(Jv),max(1,itr//50)),interval=100,repeat=False)
    ax.scatter(W[0],B[0],Jv[0],color='green',s=5,label='start point')
    ax.text(W[0],B[0],Jv[0],'start point',color='green')
    ax.scatter(W[-1],B[-1],Jv[-1],color='green',s=5,label='end point')
    ax.text(W[-1],B[-1],Jv[-1],'end point',color='green')
    ax.scatter(p,q,z,color='black',s=5,label='True minimum point',marker='*')
    ax.text(p,q,z,'True minimum',color='black')
    plt.figtext(0.02, 0.5,f"Global Minima: ({p:.4f}, {q:.4f})\nMinimum Cost: {z:.6f}")
    ax.set_xlabel("W")
    ax.set_ylabel("B")
    ax.set_zlabel("Cost")
    ax.legend()
    return fig,ani
if __name__ == "__main__":
    w=float(input("Enter value of a parameter of linear regression(w): "))
    b=float(input("Enter value of a parameter of linear regression(b): "))
    a=float(input("Enter learning rate of linear regression(a): "))
    itr=int(input("Enter number of iterations: "))
    fig,ani=run_visualizer(w,b,a,itr)
    plt.show()