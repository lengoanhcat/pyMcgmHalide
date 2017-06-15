from halide import Expr,Param,UInt

width, height, noChn = Expr("width"), Expr("height"), Expr("noChn")
noFrm = Param(UInt(16),"noFrm")
