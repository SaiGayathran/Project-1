import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Model import wf,bf,z_normalize
from visualizer import run_visualizer
plt.rcParams['animation.embed_limit'] = 100
st.set_page_config(page_title="Regression Model",page_icon="🎮",layout="wide")
st.sidebar.title("REGRESSION MODEL")
page = st.sidebar.radio("",["FPS Predictor","Gradient Descent Visualizer"])
if page == "FPS Predictor":
    st.title("FPS Predictor")
    st.caption("Predict your gaming performance")
    b11,b12,b13 = st.columns(3)
    with b11:
        cpucore = st.text_input("CPU Cores",placeholder="Enter number of cores in CPU")
    with b12:
        cputhreads = st.text_input("CPU Threads",placeholder="Enter number of threads in CPU")
    with b13:
        cputurboclk = st.text_input("CPU Turbo Clock",placeholder="Enter CPU turbo clock speed")
    b21,b22,b23=st.columns(3)
    with b21:
        gpubstclk = st.text_input("GPU Boost Clock",placeholder="Enter GPU boost clock speed")
    with b22:
        gpumemo = st.text_input("GPU Memory",placeholder="Enter GPU memory in GB")
    with b23:
        gpushdun = st.text_input("GPU Shading Units",placeholder="Enter shading units in GPU")
    gamests = st.selectbox("Quality Settings",["Low","Medium","High","Ultra"])
    predict = st.button("Predict FPS")
    quality_map = {"low": 0.25,"medium": 0.50,"high": 0.75,"ultra": 1.00}
    settings = quality_map[gamests.lower()]
    if(predict):
        cpucore = int(cpucore)
        cputhreads = int(cputhreads)
        cputurboclk = float(cputurboclk)
        gpubstclk = float(gpubstclk)
        gpumemo = int(gpumemo)
        gpushdun = int(gpushdun)
        x=np.array([cpucore,cputhreads,cputurboclk,gpubstclk,gpumemo,gpushdun,settings,cpucore**2,cputhreads**2,cputurboclk**2,gpubstclk**2,gpumemo**2,gpushdun**2,settings**2,cpucore*gpushdun,    gpumemo*gpushdun,cputurboclk*gpubstclk,gpubstclk*gpushdun,cpucore*gpubstclk])
        train,mu,sigma = z_normalize(x)
        y=np.dot(wf,train)+bf
        st.divider()
        l,c,r = st.columns([1,2,1])
        with c:
            st.caption("PREDICTED FRAME RATE")
            st.markdown(f"<h1 style='text-align: center; font-size:64px;'>{y:.0f} FPS</h1>",unsafe_allow_html=True)
else:
    st.title("Gradient Descent Visualizer")
    v11,v12,v13,v14 = st.columns(4)
    with v11:
        w = st.text_input("W",placeholder="Enter value of w")
    with v12:
            b = st.text_input("B",placeholder="Enter value of b")
    with v11:
            a = st.text_input("Alpha",placeholder="Enter learning rate")
    with v11:
            i = st.text_input("Iterations",placeholder="Enter number of iterations")
    run = st.button("Run Visualizer")
    if run:
        try:
            w = float(w)
            b = float(b)
            a = float(a)
            i = int(i)
        except ValueError:
            st.error("Enter numbers only")
            st.stop()
        fig, ani = run_visualizer(w, b, a, i)
        with st.spinner("Rendering animation..."):
            html = ani.to_jshtml()
            st.iframe(html, height=750)
        plt.close(fig)
