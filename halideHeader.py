from halide import Var,Param,UInt,Expr,cast,Int
from numpy import pi

x, y, c, t = Var("x"), Var("y"), Var("c"), Var("t")
x_outer,y_outer,x_inner,y_inner = Var("x_outer"), Var("y_outer"), Var("x_inner"), Var("y_inner")
tile_index = Var("tile_index")
M_PI = pi
