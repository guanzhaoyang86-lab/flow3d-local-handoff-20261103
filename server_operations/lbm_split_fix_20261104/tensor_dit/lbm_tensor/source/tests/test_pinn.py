"""Analytic derivative checks for the physical steady PINN field."""
import pytest
import torch
from torch import nn
from flow_baselines.pinn import PINNField

BOUNDS = torch.tensor([[-2., 4.], [1., 5.], [-3., 7.]], dtype=torch.float64)

def make_field(**overrides):
    options = dict(bounds=BOUNDS, spatial_shape=(3,4,5), hidden=8, layers=2,
                   viscosity=.13, collocation_points=7, seed=31)
    options.update(overrides)
    return PINNField(**options)

class PolynomialNetwork(nn.Module):
    def forward(self, q):
        a,b,c = q.unbind(dim=1)
        return torch.stack((a.square(),b,.5*c.square(),a+b.square()+2*c),dim=1)

def test_constant_flow_zero_residual_and_gauged_pressure():
    class ConstantNetwork(nn.Module):
        def forward(self,q):
            return q.new_tensor([1.,-2.,3.,8.]).expand(len(q),4)
    field=make_field(); field.network=ConstantNetwork()
    assert field().shape==(1,3,3,4,5)
    torch.testing.assert_close(field()[0,:,1,2,3],BOUNDS.new_tensor([1,-2,3]))
    torch.testing.assert_close(field.query(BOUNDS.mean(dim=1)[None])[:,3],BOUNDS.new_zeros(1))
    with torch.no_grad():
        losses=field.physics_loss()
    assert losses['divergence'].item()==losses['momentum'].item()==0

def test_polynomial_residual_and_physical_coordinate_scaling():
    field=make_field(); field.network=PolynomialNetwork()
    q=2*(field.collocation_coords-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0])-1
    a,b,c=q.unbind(dim=1); sx,sy,sz=2/(BOUNDS[:,1]-BOUNDS[:,0])
    divergence=2*a*sx+sy+c*sz
    momentum=torch.stack((2*a.pow(3)*sx+sx-field.viscosity*2*sx.square(),
                          3*b*sy,.5*c.pow(3)*sz+2*sz-field.viscosity*sz.square()),dim=1)
    losses=field.physics_loss()
    torch.testing.assert_close(losses['divergence'],divergence.square().mean(),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(losses['momentum'],momentum.square().sum(dim=1).mean(),atol=1e-12,rtol=1e-12)

def test_off_diagonal_convection():
    class ShearNetwork(nn.Module):
        def forward(self,q):
            xyz=BOUNDS[:,0]+(q+1)*(BOUNDS[:,1]-BOUNDS[:,0])/2
            x,y,z=xyz.unbind(dim=1)
            return torch.stack((y,x,z*0,z*0),dim=1)
    field=make_field(); field.network=ShearNetwork()
    losses=field.physics_loss()
    expected=field.collocation_coords[:,:2].square().sum(dim=1).mean()
    torch.testing.assert_close(losses['divergence'],expected.new_zeros(()))
    torch.testing.assert_close(losses['momentum'],expected)

def test_grid_xyz_components_and_zyx_storage():
    field=make_field(); field.network=PolynomialNetwork(); grid=field()
    torch.testing.assert_close(grid[0,0,1,0],BOUNDS.new_tensor([1,.25,0,.25,1]))
    torch.testing.assert_close(grid[0,1,0,:,2],torch.linspace(-1,1,4,dtype=torch.float64))
    torch.testing.assert_close(grid[0,2,:,1,2],BOUNDS.new_tensor([.5,0,.5]))

def test_parameter_gradient_matches_finite_difference():
    field=make_field(collocation_points=3)
    losses=field.physics_loss(); sum(losses.values()).backward()
    parameter=field.network[0].weight
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert parameter.grad.norm().item()>0
    analytic=parameter.grad[0,1].item(); h=1e-6; original=parameter[0,1].item()
    energies=[]
    for perturbation in (h,-h):
        with torch.no_grad():
            parameter[0,1]=original+perturbation
        energies.append(sum(field.physics_loss().values()).item())
    with torch.no_grad():
        parameter[0,1]=original
    assert analytic==pytest.approx((energies[0]-energies[1])/(2*h),rel=3e-6,abs=1e-7)

def test_seed_is_local_and_collocation_inside_domain():
    torch.manual_seed(1234); before=torch.random.get_rng_state().clone()
    first,second=make_field(),make_field()
    assert torch.equal(before,torch.random.get_rng_state())
    assert torch.equal(first.collocation_coords,second.collocation_coords)
    assert torch.equal(first(),second())
    assert bool((first.collocation_coords>=BOUNDS[:,0]).all())
    assert bool((first.collocation_coords<=BOUNDS[:,1]).all())
    assert not torch.equal(first.collocation_coords,make_field(seed=32).collocation_coords)

@pytest.mark.parametrize('change',[
    {'bounds':[[0,0],[0,1],[0,1]]},
    {'bounds':[[0,float('nan')],[0,1],[0,1]]},
    {'spatial_shape':(1,3,3)},{'viscosity':-.1},{'collocation_points':0},
])
def test_invalid_domain_and_physics_rejected(change):
    with pytest.raises(ValueError):
        make_field(**change)
