#!/usr/bin/python3
from halideHeader import *
import globalVariable as gVar

from halide import Image,Expr,Func,Float,RDom,clamp,Tuple,ImageParam
# import numpy as np
# from numpy import exp,log,sqrt,power,pi,floor
# from math import factorial

from colorDerivative import *
from temporalDerivative import *
from spatialDerivative import *
from opticalFlowEstimate import *

def main():

    orders = [5,2,2,2]
    angle = 24
    input = ImageParam(Float(32),4,"input")
    filterthreshold = Param(Float(32),"filterthreshold")
    divisionthreshold = Param(Float(32),"divisionthreshold")
    divisionthreshold2 = Param(Float(32),"divisionthreshold2")
    speedthreshold = Param(Float(32),"speedthreshold")

    gVar.width = input.width()
    gVar.height = input.height()
    gVar.noChn = input.channels()

    # Define input function to pre-process input image
    input_func = Func("input_func")
    input_func = constant_exterior(input,Expr(0))

    # Scheduling input_func

    # Define color derivative function
    d_cspec = Func("d_cspec")
    d_cspec = color_derivative(input_func)

    # Define temporal derivative
    Tx = temporal_derivative(d_cspec)
    Tx.parallel(t).vectorize(x,4).compute_root()

    # Define spatial derivative
    stBasis = Func("stBasis")
    stBasis = spatial_derivative(Tx);
    stBasis.parallel(t).vectorize(x,4).compute_root()

    # Compute velocity
    optFlw = Func("optFlw")
    optFlw = opticalFlow_estimate(stBasis,angle,orders,filterthreshold,divisionthreshold,divisionthreshold2)
    optFlw.parallel(t).vectorize(x,4).compute_root()

    # Define velocity computation
    args = ArgumentsVector()
    args.append(input)
    args.append(gVar.noFrm)
    args.append(filterthreshold)
    args.append(divisionthreshold)
    args.append(divisionthreshold2)
    args.append(speedthreshold)
    # d_cspec.compile_to_file("mcgmOpticalFlow_halide",args)
    # Tx.compile_to_file("mcgmOpticalFlow_halide",args)
    # stBasis.compile_to_file("mcgmOpticalFlow_halide",args)
    # optFlw.compile_to_c("mcgmOpticalFlow_halide",args,"mcgmOpticalFlow_halide.c")
    optFlw.compile_to_file("mcgmOpticalFlow_halide",args)

    print("mcgmOpticalFlow_halide pipeline compiled, but not yet run")

    return 0

if __name__ == "__main__":
    main()
