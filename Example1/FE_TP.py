from dolfin import *
import numpy as np
import sys
import mshr

parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(False)
parameters["reorder_dofs_serial"] = False

solver1 = KrylovSolver('gmres','ilu')
solver1.parameters['relative_tolerance'] =  1E-8
solver1.parameters["absolute_tolerance"] = 1E-8
solver1.parameters['divergence_limit'] = 1000.0
solver1.parameters['maximum_iterations'] = 10000
solver1.parameters["nonzero_initial_guess"] = False

Time_Total = Timer()
Time_Total.start()
# Create the finite element mesh for the problem
G1 = mshr.Circle(Point(0.0,0.0),0.5)
G2 = mshr.Rectangle(Point(-1.0,-1.0),Point(1.0,1.0))
G2 = G2 - G1
Geom = G2 + G1
# set subdomains
Geom.set_subdomain(1,G1)
Geom.set_subdomain(2,G2)
# create mesh with subdomain label
mesh = mshr.generate_mesh(Geom,40)
cell_domains = MeshFunction('size_t',mesh,2,mesh.domains())
print('The generated mesh has %g vertices.'%mesh.num_vertices())
# Define the facet domain for surface integral
mesh.init()
facet_domains =  MeshFunction("size_t",mesh,1)
facet_domains.set_all(0)
connf2c = mesh.topology()(1, 2)
for facet_no in range(mesh.num_facets()):
    if len(connf2c(facet_no)) == 2:
            # For each interior facet, check its two neighbor triangles	
            cell0 = connf2c(facet_no)[0]	
            cell1 = connf2c(facet_no)[1]
            # If these two cells do not have the same subdomain marks, then this facet must be on the interface
            if cell_domains.array()[cell0] != cell_domains.array()[cell1]:	
                facet_domains.array()[facet_no] = 3
            # Otherwise this facet is not on the interface, 
            # then its label must be the same as the ones of its two neighbor cells
            else: 
                facet_domains.array()[facet_no] = cell0
    elif len(connf2c(facet_no)) == 1:
            # For each interior facet, it has two neighbor cells
            # while for the one on the boundary, it has only one	
            facet_domains[facet_no] = 5
    else:
            print(facet_no,len(connf2c(facet_no)))
            sys.exit(1)


# Define the problem and solve
alpha_1 = 1.0
alpha_2 = 1.0
dx = Measure("dx",domain=mesh,subdomain_data=cell_domains)
dS = Measure('dS')[facet_domains]
V = FunctionSpace(mesh,'CG',1)
V2 = VectorFunctionSpace(mesh, 'CG', 1, dim = 2)
u = TrialFunction(V)
v = TestFunction(V)
# RHS and derivative_jump function on the interface
Expression4F = Expression('sin(x[0])+sin(x[1])',degree=2)
F_RHS = interpolate(Expression4F,V)
Expression4DJ = Expression(('-2*x[0]/(x[0]*x[0]+x[1]*x[1])',\
                            '-2*x[1]/(x[0]*x[0]+x[1]*x[1])'),degree=2)
Derivative_Jump = interpolate(Expression4DJ,V2)
# Analytical solution
Expression4u = Expression('x[0]*x[0]+x[1]*x[1]>0.5*0.5?-std::log(x[0]*x[0]+x[1]*x[1])+sin(x[0])+sin(x[1]):-std::log(0.5*0.5)+sin(x[0])+sin(x[1])',degree=2)
u_True = interpolate(Expression4u,V)
BC_Diri= DirichletBC(V, u_True, DomainBoundary())

UnitNormal = FacetNormal(mesh)
A_bilinear = alpha_1*inner(grad(u),grad(v))*dx(1)+alpha_2*inner(grad(u),grad(v))*dx(2)
L_linear = F_RHS*v*dx(1) + F_RHS*v*dx(2) + inner(Derivative_Jump('+'),UnitNormal('+'))*v('+')*dS(3)
A = assemble(A_bilinear)
b = assemble(L_linear)

BC_Diri.apply(A)
BC_Diri.apply(b)
u = Function(V)
solver1.solve(A,u.vector(),b)

Time_Total.stop()

print('The total time is %g seconds.'%Time_Total.elapsed()[0])

# Compute l2 error on the uniform points
mesh_test = UnitSquareMesh(999,999)
mesh_test.coordinates()[:] = mesh_test.coordinates()*2 - 1.0
V_test = FunctionSpace(mesh_test,'CG',1)
u_true4test = interpolate(Expression4u,V_test)
u4test = interpolate(u,V_test)
l2_error_rel = np.linalg.norm(u4test.vector().get_local() - u_true4test.vector().get_local())/np.linalg.norm(u_true4test.vector().get_local())
print ('The relative l2 error is %g'%l2_error_rel)
max_error_rel = np.linalg.norm(u4test.vector().get_local() - u_true4test.vector().get_local(),np.inf)/np.linalg.norm(u_true4test.vector().get_local(),np.inf)
print ('The relative max error is %g'%max_error_rel)


