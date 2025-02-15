from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import QuadrangleMesh,TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import LinearElasticIntegrator
from fealpy.fem import VectorSourceIntegrator, ConstIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.decorator import cartesian
from fealpy.solver import cg, spsolve
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.material.elastico_plastic_material import PlasticMaterial
from fealpy.fem.elasticoplastic_integrator import TransitionElasticIntegrator 
from fealpy.sparse import COOTensor
import argparse
# 平面应变问题
class CantileverBeamData2D():
    def domain(self):
        # 定义悬臂梁的几何尺寸
        return [0, 1, 0, 0.25]  # 长度为1 dm，高度为0.25 dm
    
    def durtion(self):
        # 时间区间
        return [0, 1]
    @cartesian
    def source(self, points: TensorLike, index=None, time: float = 0.0) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        # 假设载荷随时间变化，t为当前时间步
        val[..., 1] = -1.7
        return val
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        # 固定端边界条件（假设左端固定）
        x = points[..., 0]
        fixed_mask = x < 1e-10  # 左端固定
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[fixed_mask] = [0, 0]  # 位移为零
        
        return val
def save_results(mesh, uh, equivalent_plastic_strain, step):
    """保存VTK格式的结果文件，step从1开始计数"""
    # 转换位移场为节点数据
    displacement = uh.reshape(-1, 2)
    
    # 修正为等效塑性应变计算
    peeq = bm.sqrt(2/3 * bm.einsum('...i,...i', plastic_strain, plastic_strain))
    
            
    # 设置网格数据
    mesh.nodedata['displacement'] = displacement
    mesh.celldata['PEEQ'] = peeq.mean(axis=1)  # 单元平均
    # 设置文件名
    output_path = f"./results/step_{step:03d}.vtu"
    
    # 输出VTK文件
    mesh.to_vtk(fname=output_path)
    print(f"Saved results to {output_path}")


parser = argparse.ArgumentParser(description="Solve linear elasticity problems in arbitrary order Lagrange finite element space on QuadrangleMesh.")
parser.add_argument('--backend',
                    choices=('numpy', 'pytorch'), 
                    default='numpy', type=str,
                    help='Specify the backend type for computation, default is pytorch.')
parser.add_argument('--solver',
                    choices=('cg', 'spsolve'),
                    default='cg', type=str,
                    help='Specify the solver type for solving the linear system, default is "cg".')
parser.add_argument('--degree', 
                    default=1, type=int, 
                    help='Degree of the Lagrange finite element space, default is 2.')
parser.add_argument('--nx', 
                    default=20, type=int, 
                    help='Initial number of grid cells in the x direction, default is 4.')
parser.add_argument('--ny',
                    default=20, type=int,
                    help='Initial number of grid cells in the y direction, default is 4.')
args = parser.parse_args()
bm.set_backend(args.backend)
pde = CantileverBeamData2D()
nx, ny = args.nx, args.ny
extent = pde.domain()
mesh = TriangleMesh.from_box(box=extent, nx=nx, ny=ny)
node = mesh.entity('node')
kwargs = bm.context(node)
p = args.degree
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
gdof = space.number_of_global_dofs()
pfcm = LinearElasticMaterial(name='E1nu0.3', 
                                            elastic_modulus=1e5, poisson_ratio=0.3, 
                                            hypo='plane_stress', device=bm.get_device(mesh))
qf = mesh.quadrature_formula(q=tensor_space.p+3)
bcs, ws = qf.get_quadrature_points_and_weights()
max_increment = 20
tol = 1e-6
max_iter = 10
load_factor = 0.0
load_max = 1.0
delta_lambda = 0.05
# 材料参数
E = 1e5
nu = 0.3
sigma_y = 50  # 屈服应力
# 初始化历史变量
NC = mesh.number_of_cells()
NQ = len(ws)
plastic_strain = bm.zeros((NC, NQ, 3),**kwargs)
equivalent_plastic_strain = bm.zeros((NC, NQ),**kwargs)
D_ep_global = pfcm.elastic_matrix()

import os
os.makedirs("./results", exist_ok=True)
# 增量加载主循环
for increment in range(max_increment):
    load_factor = min(load_factor + delta_lambda, load_max)
    print(f"=== Increment {increment+1}, Load: {load_factor*100:.1f}% ===")
    
    # 牛顿迭代
    uh = tensor_space.function()
    converged = False
    
    for iter in range(max_iter):
        # 组装系统
        elasticintegrator= TransitionElasticIntegrator(D_ep_global, material=pfcm,space=tensor_space, 
                                    q=tensor_space.p+3, equivalent_plastic_strain=equivalent_plastic_strain)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(elasticintegrator)
        K = bform.assembly(format='csr')
    
        # 载荷计算_k
        @cartesian
        def loading(p):
            return load_factor * pde.source(p)
        lform = LinearForm(tensor_space) 
        lform.add_integrator(VectorSourceIntegrator(loading))
        F_ext = lform.assembly()
        
        # 计算残差
        lform = LinearForm(tensor_space) 
        cell2tdof = tensor_space.cell_to_dof()
        lform.add_integrator(ConstIntegrator(elasticintegrator.compute_internal_force(uh=uh,plastic_strain=plastic_strain),cell2tdof))
        F_int = lform.assembly()
        R = F_ext - F_int
        
        # 边界条件处理
        dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')
        K, R = dbc.apply(K, R)
        
        # 求解线性系统
        du = cg(K, R, maxiter=1000, atol=1e-14, rtol=1e-14)
        uh += du
        
        # 本构积分更新
        yield_stress = sigma_y
        success, plastic_strain, D_ep_global,equivalent_plastic_strain = elasticintegrator.constitutive_update(
            uh, plastic_strain, pfcm,yield_stress=yield_stress)
        if not success:
            break
            
        # 收敛判断
        norm = bm.linalg.norm(du)
        if norm < tol:
            converged = True
            print(f"Converged in {iter+1} iterations")
            break
            
    if not converged:
        delta_lambda *= 0.5
        print(f"Reducing step size to {delta_lambda:.3f}")
        continue
    # 保存结果
    save_results(mesh, uh, equivalent_plastic_strain, increment+1)  # 从1开始编号

    if load_factor >= load_max:
        break