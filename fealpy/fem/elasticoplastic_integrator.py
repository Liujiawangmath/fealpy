from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh, SimplexMesh, TensorMesh
from ..functionspace.space import FunctionSpace as _FS
from ..functionspace.tensor_space import TensorFunctionSpace as _TS
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod
)
from fealpy.fem.utils import SymbolicIntegration
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.linear_form import LinearForm

class TransitionElasticIntegrator(LinearElasticIntegrator):
    def __init__(self, D_ep, space, material, q, equivalent_plastic_strain, method=None):
        # 传递 method 参数并调用父类构造函数
        super().__init__(material, q, method=method)
        self.D_ep = D_ep  # 弹塑性材料矩阵
        self.space = space  # 函数空间
        self.equivalent_plastic_strain = equivalent_plastic_strain  # 等效塑性应变

    def compute_internal_force(self, uh, plastic_strain,index=_FS) -> TensorLike:
        """计算考虑塑性应变的内部力"""
        space = self.space
        mesh = space.mesh
        NC = mesh.number_of_cells()
        NQ = self.D_ep.shape[1]
        node = mesh.entity('node')
        kwargs = bm.context(node)

        # 获取单元局部位移
        cell2dof = space.cell_to_dof()
        uh = bm.array(uh,**kwargs)  
        tldof = space.number_of_local_dofs()
        uh_cell = bm.zeros((NC, tldof)) # (NC, tldof)
        for c in range(NC):
            uh_cell[c] = uh[cell2dof[c]]
        qf = mesh.quadrature_formula(q=space.p+3)   
        bcs, ws = qf.get_quadrature_points_and_weights()
        # 计算应变
        B = self.material.strain_matrix(True, gphi=space.grad_basis(bcs))
        strain_total = bm.einsum('cqijk,ci->cqj', B, uh_cell)
        strain_elastic = strain_total - plastic_strain

        # 计算应力
        stress = bm.einsum('cqij,cqj->cqi', self.D_ep, strain_elastic)

        # 组装内部力
        cm = mesh.entity_measure('cell')
        F_int = bm.zeros_like(uh, **kwargs)
        F_int_cell = bm.einsum('q, c, cqijk,cqj->ci', 
                             ws, cm, B, stress) # (NC, tdof)
        
        return F_int_cell


    def constitutive_update(self, uh, plastic_strain_old, material,yield_stress):
        """执行本构积分返回更新后的状态"""
        # 计算试应变
        space = self.space
        mesh = space.mesh
        node = mesh.entity('node')
        kwargs = bm.context(node)
        qf = mesh.quadrature_formula(q=space.p+3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        B = material.strain_matrix(True,gphi=space.grad_basis(bcs))
        uh = bm.array(uh,**kwargs)  
        tldof = space.number_of_local_dofs()
        NC = mesh.number_of_cells() 
        uh_cell = bm.zeros((NC, tldof),**kwargs) # (NC, tldof)
        cell2dof = space.cell_to_dof()
        for c in range(NC):
            uh_cell[c] = uh[cell2dof[c]]
        strain_total = bm.einsum('cqijk,ci->cqj', B, uh_cell)
        strain_trial = strain_total - plastic_strain_old
        # 弹性预测
        stress_trial = bm.einsum('cqij,cqi->cqj', material.elastic_matrix(), strain_trial)
        
        # 屈服判断
        s_trial = stress_trial - bm.mean(stress_trial[..., :2], axis=-1, keepdims=True)
        sigma_eff = bm.sqrt(3/2 * bm.einsum('...i,...i', s_trial, s_trial))
        yield_mask = sigma_eff > yield_stress
        
        # 塑性修正
        if bm.any(yield_mask):
            # 计算流动方向
            n = s_trial / (sigma_eff[..., None] + 1e-12)
            
            # 计算塑性乘子
            delta_gamma = (sigma_eff[yield_mask] - yield_stress) / (3*material.mu)
            # 计算等效塑性应变增量
            delta_peeq = delta_gamma * bm.sqrt(2/3)

            # 累积到全局变量
            self.equivalent_plastic_strain[yield_mask] += delta_peeq

            # 更新塑性应变
            plastic_strain_new = plastic_strain_old.copy()
            plastic_strain_new[yield_mask] += delta_gamma[:, None] * n[yield_mask]
            
            # 更新弹塑性矩阵
            D_ep = self.update_elastoplastic_matrix(material, n, sigma_eff, yield_mask)
             # 在更新D_ep后添加
            eigenvalues = bm.linalg.eigvalsh(D_ep)
            print("Max eigenvalue:", eigenvalues.max())
            return True, plastic_strain_new, D_ep, self.equivalent_plastic_strain
        else:
            return True, plastic_strain_old, material.elastic_matrix(),self.equivalent_plastic_strain

    def update_elastoplastic_matrix(self, material, n, sigma_eff, yield_mask):
        """正确的弹塑性矩阵构造"""
        # 获取弹性矩阵
        D_e = material.elastic_matrix()  # (..., 3, 3)
        # 计算分母项 H = n:D_e:n (标量)
        H = bm.einsum('...i,...ij,...j->...', n, D_e, n)
        H = H[..., None, None]
        # 计算塑性修正项
        numerator = bm.einsum('...i,...j->...ij', 
                            bm.einsum('...ik,...k', D_e, n),
                            bm.einsum('...jk,...k', D_e, n))
        
        # 理想塑性时 H'=0
        D_ep = D_e - numerator / (H + 1e-12)
        
        return bm.where(yield_mask[..., None, None], D_ep, D_e)

            

    def assembly(self, space: _FS) -> TensorLike:
        '''组装切线刚度矩阵'''
        mesh = getattr(space, 'mesh', None)
        D_ep = self.D_ep
        cm, ws, detJ, D, B = self.fetch_voigt_assembly(space)
        
        if isinstance(mesh, TensorMesh):
            KK = bm.einsum('c, cq, cqki, cqkl, cqlj -> cij',
                            ws, detJ, B, D_ep, B)
        else:
            KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij',
                            ws, cm, B, D_ep, B)
        
        return KK # (NC, tdof, tdof)