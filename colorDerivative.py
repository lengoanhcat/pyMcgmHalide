from halide import Func,Image,RDom,Float,cast,UInt,sum
from halideHeader import *

def color_derivative(input):
    m = 1.0/3.0

    specDeri = Func("specDeri")

    colorRF = Image(type=Float(32),x=3,y=3,name="colorRF")
    colorRF[0,0]=1.0/3.0; colorRF[1,0]=1.0/3.0; colorRF[2,0]=1.0/3.0
    colorRF[0,1]=0.25; colorRF[1,1]=0.25; colorRF[2,1]=-0.5
    colorRF[0,2]=0.5; colorRF[1,2]=-0.5; colorRF[2,2]=0.0

    rf_x = RDom(0,3)

    specDeri[x,y,c,t] = sum(rf_x,input[x,y,rf_x,t]*colorRF[rf_x,c],"sum_colorRF")

    return specDeri
