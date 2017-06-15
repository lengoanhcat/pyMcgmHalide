from halide import Image,Expr,Func,Float,cast,RDom,clamp,sum,Tuple
from numpy import exp,log,sqrt,power,pi,zeros
from halideHeader import *
import globalVariable as gVar

def temporal_derivative(input):
    tn = 23; alpha = 10.0; tau = 0.25
    d_ltm0 = Image(type=Float(32),x=23,name="d_ltm0"); d_ltm0[0] = 0.0;
    d_ltm1 = Image(type=Float(32),x=23,name="d_ltm1"); d_ltm1[0] = 0.0;
    d_ltm2 = Image(type=Float(32),x=23,name="d_ltm2"); d_ltm2[0] = 0.0;
    # d_ltm0 = d_ltm1 = d_ltm2 = zeros(23,dtype=float)

    for rt in range(1,tn):
        der_0rd = exp(-power(log(rt/alpha)/tau,2.0))/(sqrt(M_PI)*alpha*tau*exp((power(tau,2.0)/4.0)))
        d_ltm0[rt] = der_0rd
        der_1st = -2.0*((log(rt/alpha))/(float(power(tau,2.0))*rt))*der_0rd
        d_ltm1[rt] = der_1st
        d_ltm2[rt] = -2.0*((log(rt/alpha))/(float(power(tau,2.0))*rt))*der_1st - (2.0/(power(tau*rt,2.0))) * (1.0-log(rt/alpha)) * der_0rd

    rt = RDom(0,tn)

    ####################################################################
    # Apply derivative filter along temporal domain of input sequences #
    ####################################################################

    T = Func("T")
    t_clamped = clamp(t,0,gVar.noFrm-1)
    T_clamped = Func("T_clamped")
    T_clamped[x,y,c,t] = input[x,y,c,t_clamped]

    T_expr1 = sum(rt,d_ltm0[rt.x]*T_clamped[x,y,c,t+rt.x],"sum_T0")
    T_expr2 = sum(rt,d_ltm1[rt.x]*T_clamped[x,y,c,t+rt.x],"sum_T1")
    T_expr3 = sum(rt,d_ltm2[rt.x]*T_clamped[x,y,c,t+rt.x],"sum_T2")
    T[x,y,c,t] = Tuple(T_expr1,T_expr2,T_expr3)
    return T
