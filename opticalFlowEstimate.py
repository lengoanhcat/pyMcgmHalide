from halide import *
import numpy
from numpy import power,cos,sin,pi
from math_utils import *
from halideHeader import *
import globalVariable as gVar

def Mgetfilterindex(x_order: numpy.uint8,y_order: numpy.uint8,t_order: numpy.uint8,numSTB: numpy.uint8,numSB: numpy.uint8) -> numpy.int:
    # Get filter index with respect to (x,y,t) order
    index = numpy.int((x_order+y_order)*(x_order+y_order+1)/2 + y_order)
    if (index >= numSB):
        index = -64

    index = index + t_order * numSB
    assert(index<numSTB)
    return index

def ColorMgetfilter(stBasis,angle,iXo: numpy.uint8,iYo: numpy.uint8,iTo: numpy.uint8,iCo: numpy.uint8):

    # Compute a rotated basis at (iXo,iYo,iTo,iCo) order with angle value
    numSTB = 63
    numSB = 21
    angle = -1*angle - M_PI/2

    work = Func("work") # work: rotated basis at a particular spatio-temporal order
    work[x,y,t] = 0.0

    weights = numpy.zeros(iXo+iYo+1,dtype=float)

    # compute weights for possible orders
    for i in range(0,iXo+1):
        for j in range(0,iYo+1):
            weights[iXo+iYo-i-j] += combination(iXo,i)*combination(iYo,j)*power(-1.0,i)*power(cos(angle),iXo-i+j)*power(sin(angle),iYo+i-j)

    # get filtered expression at particular order and angle value
    for k in range(0,iXo+iYo):
        index = Mgetfilterindex(iXo+iYo-k,k,iTo,numSTB,numSB)
        if (index>0 and weights[iXo+iYo-k] != 0):
            work[x,y,t] += weights[iXo+iYo-k]*stBasis[x,y,iCo,t][index]

    work.compute_root()
    return work

def ColorMgather(stBasis,angle,orders,filterthreshold,divisionthreshold,divisionthreshold2):
    x_order = orders[0]; y_order = orders[1]; t_order = orders[2]; c_order = orders[3];
    X = Func("X"); Y = Func("Y"); T = Func("T");
    Xrg = Func("Xrg"); Yrg = Func("Yrg"); Trg = Func("Trg")
    max_order = x_order

    Xk_uI = numpy.zeros((max_order,),dtype=numpy.uint8)
    Yk_uI = numpy.zeros((max_order,),dtype=numpy.uint8)
    Tk_uI = numpy.zeros((max_order,),dtype=numpy.uint8)
    Xk = list(); Yk = list(); Tk = list();

    for iO in range(0,x_order):
        Xk.append(Func()); Xk[iO][x,y,t] = 0.0
        Yk.append(Func()); Yk[iO][x,y,t] = 0.0
        Tk.append(Func()); Tk[iO][x,y,t] = 0.0

    k = 0
    for iXo in range(0,x_order):
        for iYo in range(0,y_order):
            for iTo in range(0,t_order):
                for iCo in range(0,c_order):
                    if ((iYo+iTo+iCo == 0 or iYo+iTo+iCo == 1) and (iXo+iYo+iTo+iCo+1 < x_order+1)):
                        X = ColorMgetfilter(stBasis,angle,iXo+1,iYo,iTo,iCo)
                        Y = ColorMgetfilter(stBasis,angle,iXo,iYo+1,iTo,iCo)
                        T = ColorMgetfilter(stBasis,angle,iXo,iYo,iTo+1,iCo)
                        Xrg = ColorMgetfilter(stBasis,angle,iXo+1,iYo,iTo,iCo+1)
                        Yrg = ColorMgetfilter(stBasis,angle,iXo,iYo+1,iTo,iCo+1)
                        Trg = ColorMgetfilter(stBasis,angle,iXo,iYo,iTo+1,iCo+1)
                        k = iXo + iYo + iTo + iCo
                        Xk[k][x,y,t] += X[x,y,t] + Xrg[x,y,t]
                        Yk[k][x,y,t] += Y[x,y,t] + Yrg[x,y,t]
                        Tk[k][x,y,t] += T[x,y,t] + Trg[x,y,t]

                        Xk[k].update(int(Xk_uI[k])); Xk_uI[k]=Xk_uI[k]+1;
                        Yk[k].update(int(Yk_uI[k])); Yk_uI[k]=Yk_uI[k]+1;
                        Tk[k].update(int(Tk_uI[k])); Tk_uI[k]=Tk_uI[k]+1;

    # Scheduling
    for iO in range(0,k+1):
        Xk[iO].compute_root()
        Yk[iO].compute_root()
        Tk[iO].compute_root()

    st_expr = ExprsVector()
    for i in range(0,6):
        st_expr.append(Expr(0.0))

    for iK in range(0,k+1):
        st_expr[0] += Xk[iK][x,y,t]*Tk[iK][x,y,t]
        st_expr[1] += Tk[iK][x,y,t]*Tk[iK][x,y,t]
        st_expr[2] += Xk[iK][x,y,t]*Xk[iK][x,y,t]
        st_expr[3] += Yk[iK][x,y,t]*Tk[iK][x,y,t]
        st_expr[4] += Yk[iK][x,y,t]*Yk[iK][x,y,t]
        st_expr[5] += Xk[iK][x,y,t]*Yk[iK][x,y,t]

    st = Func("st"); st[x,y,t] = Tuple(st_expr);
    st.compute_root();

    x_clamped = Expr("x_clamped"); x_clamped = clamp(x,0,gVar.width-1);
    y_clamped = Expr("y_clamped"); y_clamped = clamp(y,0,gVar.height-1);
    st_clamped = Func("st_clamped")
    st_clamped[x,y,t] = Tuple(st[x_clamped,y_clamped,t])

    win = 7;
    rMF = RDom(0,win,0,win);

    st_filtered = list();
    for iPc in range(0,6):
        # iPc: index of product component
        # Apply average filter
        st_filtered.append(Func())
        st_filtered[iPc][x,y,t] = sum(rMF,st_clamped[x+rMF.x,y+rMF.y,t][iPc]/Expr(float(win*win)),"mean_filter")
        st_filtered[iPc].compute_root()

    # tmpOut = Func("tmpOut")
    # tmpOut[x,y,t] = Tuple(st_filtered[0][x,y,t],st_filtered[1][x,y,t],st_filtered[2][x,y,t],st_filtered[3][x,y,t],st_filtered[4][x,y,t])
    # return tmpOut

    pbx = Tuple(st_filtered[2][x,y,t],st_filtered[5][x,y,t],st_filtered[0][x,y,t])
    pby = Tuple(st_filtered[5][x,y,t],st_filtered[4][x,y,t],st_filtered[3][x,y,t])
    pbt = Tuple(st_filtered[0][x,y,t],st_filtered[3][x,y,t],st_filtered[1][x,y,t])

    pbxy = Func("pbxy"); pbxy = cross(pby,pbx); pbxy.compute_root();
    pbxt = Func("pbxt"); pbxt = cross(pbx,pbt); pbxt.compute_root();
    pbyt = Func("pbyt"); pbyt = cross(pby,pbt); pbyt.compute_root();

    pbxyd = Func("pbxyd"); pbxyd = dot(pby,pbx); pbxyd.compute_root();
    pbxtd = Func("pbxtd"); pbxtd = dot(pbx,pbt); pbxtd.compute_root();
    pbytd = Func("pbytd"); pbytd = dot(pby,pbt); pbytd.compute_root();

    yt_xy = Func("yt_xy"); yt_xy = dot(pbyt[x,y,t],pbxy[x,y,t]); yt_xy.compute_root();
    xt_yt = Func("xt_yt"); xt_yt = dot(pbxt[x,y,t],pbyt[x,y,t]); xt_yt.compute_root();
    xt_xy = Func("xt_xy"); xt_xy = dot(pbxt[x,y,t],pbxy[x,y,t]); xt_xy.compute_root();
    yt_yt = Func("yt_yt"); yt_yt = dot(pbyt[x,y,t],pbyt[x,y,t]); yt_yt.compute_root();
    xt_xt = Func("xt_xt"); xt_xt = dot(pbxt[x,y,t],pbxt[x,y,t]); xt_xt.compute_root();
    xy_xy = Func("xy_xy"); xy_xy = dot(pbxy[x,y,t],pbxy[x,y,t]); xy_xy.compute_root();

    Tk_tuple = Tuple(Tk[0][x,y,t],Tk[1][x,y,t],Tk[2][x,y,t],
                           Tk[3][x,y,t],Tk[4][x,y,t]);
    Tkd = Func("Tkd"); Tkd = dot(Tk_tuple,Tk_tuple); Tkd.compute_root();

    # // Expr Dimen = pbxyd/xy_xy;
    kill = Expr(1.0);

    Oxy = Func("Oxy"); Oxy[x,y,t] = Mdefdiv(st_filtered[5][x,y,t] - Mdefdivang(yt_xy[x,y,t],yt_yt[x,y,t],pbxyd[x,y,t],divisionthreshold2)*st_filtered[3][x,y,t]*kill,st_filtered[4][x,y,t],divisionthreshold);
    Oxy.compute_root();

    Oyx = Func("Oyx"); Oyx[x,y,t] = Mdefdiv(st_filtered[5][x,y,t] + Mdefdivang(xt_xy[x,y,t],xt_xt[x,y,t],pbxyd[x,y,t],divisionthreshold2)*st_filtered[0][x,y,t]*kill,st_filtered[2][x,y,t],divisionthreshold);
    Oyx.compute_root();

    C0 = Func(); C0[x,y,t] = st_filtered[3][x,y,t] * Mdefdivang(Expr(-1.0)*xt_yt[x,y,t],yt_yt[x,y,t],pbxyd[x,y,t],divisionthreshold2)*kill
    C0.compute_root();

    M0 = Func(); M0[x,y,t] = Mdefdiv(st_filtered[0][x,y,t] + C0[x,y,t], st_filtered[1][x,y,t]*pow(Mdefdivang(xt_yt[x,y,t],yt_yt[x,y,t],pbxyd[x,y,t],divisionthreshold2),Expr(2.0)),divisionthreshold);
    M0.compute_root();

    C1 = Func(); C1[x,y,t] = st_filtered[5][x,y,t] * Mdefdivang(Expr(-1.0)*xt_xy[x,y,t],xy_xy[x,y,t],pbxyd[x,y,t],divisionthreshold2)*kill;
    C1.compute_root();

    P1 = Func(); P1[x,y,t] = pow(Mdefdivang(xt_yt[x,y,t],xt_xt[x,y,t],pbxyd[x,y,t],divisionthreshold2),Expr(2.0))*kill + 1.0;
    P1.compute_root();

    # // 4 debug
    # // Func tmpOut("tmpOut"); tmpOut[x,y,t] = Tuple(Oxy[x,y,t],Oyx[x,y,t],C0[x,y,t],M0[x,y,t],C1[x,y,t],P1[x,y,t]);
    # // return tmpOut;


    Q1 = Func(); Q1[x,y,t] = st_filtered[2][x,y,t] * (pow(Oyx[x,y,t],Expr(2.0))+Expr(1.0));
    Q1.compute_root();

    M1 = Func(); M1[x,y,t] = Mdefdiv(((st_filtered[0][x,y,t]-C1[x,y,t])*P1[x,y,t]),Q1[x,y,t],divisionthreshold);
    M1.compute_root();

    C2 = Func(); C2[x,y,t] = st_filtered[0][x,y,t] * Mdefdivang(Expr(-1.0)*xt_yt[x,y,t],xt_xt[x,y,t],pbxyd[x,y,t],divisionthreshold2)*kill;
    C2.compute_root();

    M2 = Func(); M2[x,y,t] = Mdefdiv(st_filtered[3][x,y,t]+C2[x,y,t],st_filtered[1][x,y,t]*(pow(Mdefdivang(xt_yt[x,y,t],xt_xt[x,y,t],pbxyd[x,y,t],divisionthreshold2),Expr(2.0))*kill+Expr(1.0)),divisionthreshold);
    M2.compute_root();

    C3 = Func(); C3[x,y,t] = st_filtered[5][x,y,t] * Mdefdivang(yt_xy[x,y,t],xy_xy[x,y,t],pbxyd[x,y,t],divisionthreshold2)*kill;
    C3.compute_root();

    P3 = Func(); P3[x,y,t] = pow(Mdefdivang(xt_yt[x,y,t],yt_yt[x,y,t],pbxyd[x,y,t],divisionthreshold2),Expr(2.0))*kill + Expr(1.0);
    P3.compute_root();

    Q3 = Func(); Q3[x,y,t] = st_filtered[4][x,y,t] * (pow(Oxy[x,y,t],Expr(2.0))+Expr(1.0));
    Q3.compute_root();

    M3 = Func(); M3[x,y,t] = Mdefdiv(((st_filtered[3][x,y,t]-C3[x,y,t])*P3[x,y,t]),Q3[x,y,t],divisionthreshold);
    M3.compute_root();

    basisAtAngle = Func("basisAtAngle");
    basisAtAngle[x,y,t] = Tuple(M0[x,y,t],M1[x,y,t],M2[x,y,t],M3[x,y,t],Tkd[x,y,t]);
    # basisAtAngle.compute_root();

    return basisAtAngle;

def opticalFlow_estimate (stBasis, nAngle, orders, \
                           filterthreshold, divisionthreshold,\
                           divisionthreshold2):

# This function estimates components of optical flow fields as well as its speed and direction of movement
# basis: from spatio-temporal filters, angle: number of considered angles
# orders: x (spatial index), y ( spatial index ), t (time index) and s ?
# ColorMgather function in MATLAB
 # Pipeline:
 # 1. Compute oriented filter basis at a particular angle
 # {
 #     basis -> X
 #           -> Y
 #           -> T
 #           -> Xrg
 #           -> Yrg
 #           -> Trg
 #           -> Xk
 #           -> Yk
 #           -> Tk
 # }
    Tkd = Func("Tkd");
    fA0 = Func("fA0"); fA0[x,y,t] = Expr(0.0);
    fA1 = Func("fA1"); fA1[x,y,t] = Expr(0.0);
    fA2 = Func("fA2"); fA2[x,y,t] = Expr(0.0);
    fA3 = Func("fA3"); fA3[x,y,t] = Expr(0.0);
    fD0 = Func("fD0"); fD0[x,y,t] = Expr(0.0);
    fD1 = Func("fD1"); fD1[x,y,t] = Expr(0.0);
    fD2 = Func("fD2"); fD2[x,y,t] = Expr(0.0);
    fD3 = Func("fD3"); fD3[x,y,t] = Expr(0.0);
    fN0 = Func("fN0"); fN0[x,y,t] = Expr(0.0);
    fN1 = Func("fN1"); fN1[x,y,t] = Expr(0.0);
    fN2 = Func("fN2"); fN2[x,y,t] = Expr(0.0);
    fN3 = Func("fN3"); fN3[x,y,t] = Expr(0.0);
    basisAtAngle = list();# Func basisAtAngle[nAngle/2];

# Compute spatial-temporal basis
# Error: not able to sum two basisAtAngle Exprssion / Tuple
    for iA in range(0,int(nAngle / 2)):
        aAngle = 2*iA*M_PI/nAngle;
        basisAtAngle.append(ColorMgather(stBasis, aAngle, orders, filterthreshold,
                                        divisionthreshold, divisionthreshold2));
        basisAtAngle[iA].compute_root();

        M0 = Expr();M1 = Expr(); M2 = Expr(); M3 = Expr();
        M0 = basisAtAngle[iA][x,y,t][0]; M1 = basisAtAngle[iA][x,y,t][1];
        M2 = basisAtAngle[iA][x,y,t][2]; M3 = basisAtAngle[iA][x,y,t][3];

        fD0[x,y,t] += M0 * M0;
        fD1[x,y,t] += M0 * M2;
        fD2[x,y,t] += M2 * M0;
        fD3[x,y,t] += M2 * M2;
        fN0[x,y,t] += M1 * M0;
        fN1[x,y,t] += M1 * M2;
        fN2[x,y,t] += M3 * M0;
        fN3[x,y,t] += M3 * M2;
        cosaAngle = Expr(float(cos(aAngle)));
        sinaAngle = Expr(float(sin(aAngle)));
        fA0[x,y,t] += abs(M0) * M1 * cosaAngle;
        fA1[x,y,t] += abs(M2) * M3 * sinaAngle;
        fA2[x,y,t] += abs(M0) * M1 * sinaAngle;
        fA3[x,y,t] += abs(M2) * M3 * cosaAngle;

    Tkd[x,y,t] = basisAtAngle[int(nAngle/2)-1][x,y,t][4];

    fD0.compute_root();
    fD1.compute_root();
    fD2.compute_root();
    fD3.compute_root();
    fN0.compute_root();
    fN1.compute_root();
    fN2.compute_root();
    fN3.compute_root();
    fA0.compute_root();
    fA1.compute_root();
    fA2.compute_root();
    fA3.compute_root();
    Tkd.compute_root();

    top_func = Func("top_func");
    bottom_func = Func("bottom_func");
    speed0 = Expr("speed0");
    speed1 = Expr("speed1");

# Polar fig
    top_func[x,y,t] = fN0[x,y,t] * fN3[x,y,t] - fN1[x,y,t] * fN2[x,y,t];
    top_func.compute_root();
    bottom_func[x,y,t] = fD0[x,y,t]  * fD3[x,y,t]  - fD1[x,y,t]  * fD2[x,y,t];
    bottom_func.compute_root();

    speed0 = sqrt(sqrt(abs(Mdefdiv(top_func[x,y,t] , bottom_func[x,y,t], Expr(0.0)))));
    speed1 = Manglecalc(fA0[x,y,t] , fA1[x,y,t] , fA2[x,y,t] , fA3[x,y,t]);
# Compute the results
    speed0 = select(abs(Tkd[x,y,t]) > filterthreshold,speed0,Expr(0.0));
    speed1 = select(abs(Tkd[x,y,t]) > filterthreshold,speed1,Expr(0.0));

    speed = Func("speed"); speed[x,y,t] = Tuple(speed0,speed1);
    return speed;

# //Bimg = [T0n speed0 speed1] ?
# //    Func img = outputvelocity(T0n,Func(speed0),Func(speed1),16, speedthreshold, filterthreshold);
