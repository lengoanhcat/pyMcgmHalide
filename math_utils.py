from halide import ExprsVector,Expr,Func,Tuple,select,abs,atan2
from math import factorial
from halideHeader import *

def combination(v,k):
    # calculate combinations of v elements when choose k of them
    if (v<k):
        return 0
    elif (v==k):
        return 1
    else:
        return factorial(v)/(factorial(v-k)*factorial(k))

def cross(a, b):
    # compute cross product of two vectors
    cp_expr = ExprsVector()
    for i in range(0,3):
        cp_expr.append(Expr(0.0))

    # Compute a cross-product of two vector
    cp_expr[0] = a[1]*b[2] - a[2]*b[1];
    cp_expr[1] = a[2]*b[0] - a[0]*b[2];
    cp_expr[2] = a[0]*b[1] - a[1]*b[0];

    cp_func = Func(); cp_func[x,y,t] = Tuple(cp_expr);

    return cp_func;

def dot(a, b):
    # compute dot product of two vectors
    dp = Expr("dp"); dp = 0.0;
    # assert(a.size() == b.size());
    for iC in range(0,a.size()):
        dp += a[iC]*b[iC];

    dp_func = Func(); dp_func[x,y,t] = dp;
    return dp_func;

def Mdefdiv(A, B, divth):
    # Avoid dividing by 0
    return select(abs(B)>divth,A/B,Expr(0.0));


def Mdefdivang(A, B, dot, divth):
    #  Avoid dividing by 0
    return select(abs(A/dot)>divth,A/B,Expr(0.0));

def Manglecalc(A0, A1, A2, A3):
    return atan2((A2 + A3), (A0 - A1));
