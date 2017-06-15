from halide import Image,Expr,Func,Float,cast,RDom,clamp,sum,Tuple,ExprsVector
from numpy import exp,log,sqrt,power
from math import factorial
from halideHeader import *

def spatial_derivative(T):
    # This function computes spatial gussian derivative up to nfilt-order
    nfilt = 5; numSTB = 63; numTB = 3; sn = 23; sigma = 1.5;
    SFILT = Image(type=Float(32),x=sn,y=nfilt+1)
    for rs in range(0,sn):
        si = rs - (sn-1)/2
        gauss = exp(-(power(si,2.0)/(4*sigma)))/sqrt(4*sigma*M_PI)
        for ro in range(0,nfilt+1):
            H = 0.0
            for io in range(0,int(ro/2)+1):
                H = H + power(-1,io) * (power((2*si),(ro-2*io))/power((4*sigma),(ro-io)))/float(factorial(io)*factorial(ro-2*io))
            H = factorial(ro)*H
            SFILT[rs,ro] = H*gauss

    rs = RDom(0,sn)
    ###############################################
    # Apply spatial derivative filter on T0,T1,T2 #
    ###############################################
    tmp_col= Func("tmp_col")
    tmp_col_expr = ExprsVector() # empty list of  expression: (nfilt+1)*numTB
    basis = Func("basis")
    basis_expr = ExprsVector() # empty list of expression

    iB = 0

    for iTf in range(0,numTB):
        for iSf in range(0,nfilt+1):
            tmp_col_expr.append(sum(rs,T[x,y+rs.x,c,t][iTf]*SFILT[rs.x,iSf],"sum_col"))

    tmp_col[x,y,c,t] = Tuple(tmp_col_expr)
    tmp_col.compute_root()

    # FIR filter on vertical axis
    for iTf in range(0,numTB):
        for iSf in range(0,nfilt+1):
            for iSf1 in range(0,iSf+1):
                for iSf2 in range(0,iSf+1):
                    if (iSf1+iSf2 == iSf):
                        basis_expr.append(sum(rs,tmp_col[x+rs.x,y,c,t][iTf*(nfilt+1)+iSf1]*SFILT[rs.x,iSf2],"sum_basis"))

    basis[x,y,c,t] = Tuple(basis_expr)
    return basis
