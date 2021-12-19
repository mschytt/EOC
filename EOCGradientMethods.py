# -*- coding: utf-8 -*-
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import time

#Mesh, Function space, solver initilization
mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, "CG", 1)
#V = FunctionSpace(mesh, 'P', 1)

#Dirichlet boundary condition
yb = Constant(0)
def Boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, yb, Boundary)

# Initialize problem data
x0, x1 = MeshCoordinates(mesh)

#Regularization parameter
lmbd = 1e-7
#Desired y_Omega
yd = conditional((2/10)**2<(x0-0.5)**2+(x1-0.5)**2, conditional((3/10)**2>(x0-0.5)**2+(x1-0.5)**2,1,0) , 0)
#Functional
def J(y,u,yd=yd,lmbd=lmbd):
    val = 0.5*assemble(((y-yd)**2+lmbd*(u)**2)*dx)
    return val
#Billinear form
y = TrialFunction(V)
v = TestFunction(V)
c = conditional(x0<0.5,0.5,2)
a = dot(c*grad(y), grad(v))*dx
#Source
beta = Constant(1)
def Source_term(u):
    return u*v*dx
#Box-constraints
u_lower = Constant(-100)
u_upper = Constant(100)
#Solver
def Dirichlet_Solve(a,f,bc = bc):
    sol      = Function(V)
    f_source = Source_term(f)
    solve(a == f_source, sol,bc)
    return sol


def Adjoint_state(y):
    p = Dirichlet_Solve(a,y-yd)
    return p

def Gradient_f(y,u):
    p  = Adjoint_state(y)
    df = beta*p+lmbd*u
    return df

#Armijo linesearch with Projection
def Armijo(y0,u0,df0,h, alpha = 2000, c = 1e-1 , mu = 0.5):
    f0    = J(y0,u0)
    df0h  = assemble(df0*h*dx)
    u     = u0 + alpha*h
    #Project step onto box-constraints
    u = conditional(u<u_upper,conditional(u>u_lower,u,u_lower),u_upper)
    u = project(u,V)
    y     = Dirichlet_Solve(a, beta*u)
    fnew  = J(y,u)
    while f0 + c*alpha*df0h < fnew:
        alpha*= mu
        print(alpha)
        u     = u0 + alpha*h
        #Project step onto box-constraints
        u = conditional(u<u_upper,conditional(u>u_lower,u,u_lower),u_upper)
        u = project(u,V)
        y     = Dirichlet_Solve(a, beta*u)
        fnew  = J(y,u)
    return f0, fnew, alpha

#Exact linesearch for Conditioned Gradient
def Exact_linesearch(y,yw,u,w):
    dyd = y-yd
    dys = yw-y
    dus = w-u
    g1 = assemble((dyd*dys+lmbd*u*dus)*dx)
    g2 = assemble(0.5*(dys**2+lmbd*dus**2)*dx)
    if g2 == 0:
        print('g2=0')
    gminarg = -0.5*g1/g2
    alpha = min(max(0,gminarg),1)
    print(alpha)
    return(alpha)
    
#Initial guess
u = Constant(0)
#Project guess onto box-constraints
u = conditional(u<u_upper,conditional(u>u_lower,u,u_lower),u_upper)
u = project(u,V)
y = Dirichlet_Solve(a,beta*u)
#Iterate Conditional Gradient Method
iter_max = 200
fcond_vals = np.zeros(iter_max+1)
res_vals = np.zeros(iter_max)
for i in range(iter_max):
    if i == 0:
        fcond_vals[0] = J(y,u)
    df    = Gradient_f(y,u)
    w     = conditional(df >= 0,u_lower,u_upper)
    res_vals[i] = assemble(df*(w-u)*dx)
    print(res_vals[i])
    yw     = Dirichlet_Solve(a,beta*w)
    alpha = Exact_linesearch(y,yw,u,w)
    u     = u + alpha*(w-u)
    u     = project(u,V)
    y     = Dirichlet_Solve(a,beta*u)
    fcond_vals[i+1]    = J(y,u)
    #print(f_vals[i+1])
vtkfile = File('Finalplots/controlcond.pvd')
u=project(u,V)
vtkfile << u
vtkfile = File('Finalplots/statecond.pvd')
y=project(y,V)
vtkfile << y

#Initial guess
u = Constant(0)
#Project guess onto box-constraints
u = conditional(u<u_upper,conditional(u>u_lower,u,u_lower),u_upper)
u = project(u,V)
#Iterate Projected Gradient Descent
iter_max = 200
fproj_vals = np.zeros(iter_max+1)
for i in range(iter_max):
    y     = Dirichlet_Solve(a,beta*u)
    if i == 0:
        fproj_vals[0] = J(y,u)
    df    = Gradient_f(y,u)
    h     = -df
    f0, fnew,alpha = Armijo(y,u,df,h)
    print(alpha)
    fproj_vals[i+1]    = fnew
    u     = u + alpha*h
    u     = conditional(u<u_upper,conditional(u>u_lower,u,u_lower),u_upper)
    u     = project(u,V)
    print(i)
    #print(fnew,f0)
vtkfile = File('Finalplots/projcontrol.pvd')
u=project(u,V)
vtkfile << u
vtkfile = File('Finalplots/projstate.pvd')
y=project(y,V)
vtkfile << y
