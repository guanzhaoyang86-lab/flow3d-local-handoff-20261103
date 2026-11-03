import torch
import runtime as r

def test_real_dit_partial_support_forward_backward():
    torch.manual_seed(31)
    model = r.build_denoiser({'architecture': 'dit3d', 'condition_dim': 1,
        'spatial_shape': [4, 4, 4], 'dit_hidden_dim': 24, 'dit_depth': 1,
        'dit_heads': 3, 'patch_size': 2})
    diffusion = r.GaussianDiffusion(num_steps=10, prediction_type='v', schedule='cosine', clip_x0=None)
    target = torch.randn(1, 3, 4, 4, 4)
    support = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    support[:, 1:3, 1:3, 1:3] = True
    padded = torch.where(support[:, None], target, torch.full_like(target, float('nan')))
    loss = r.partial_loss(model, diffusion, padded, support, torch.Generator().manual_seed(47))
    other = r.partial_loss(model, diffusion, target, support, torch.Generator().manual_seed(47))
    assert torch.equal(loss, other) and torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    before = [p.detach().clone() for p in model.parameters()]
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

def test_statistics_fit_uses_reference_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Prior statistics accessed observations')
    monkeypatch.setattr(r.physics, 'load_observation', forbidden)
    field = torch.arange(576, dtype=torch.float32).reshape(3, 3, 4, 4, 4)
    support = torch.ones(3, 4, 4, 4, dtype=torch.bool)
    monkeypatch.setattr(r.physics, 'load_reference', lambda row: (field, support, torch.tensor([0., .1, .3])))
    stats = r.fit_statistics([{'case_id': 'train_synthetic'}])
    assert stats['fit_split'] == 'train' and stats['training_case_ids'] == ['train_synthetic']
    assert len(stats['mean']) == 3 and min(stats['std']) > 0
