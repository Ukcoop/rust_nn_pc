use tch::{Tensor, nn};

/// A lightweight per-feature attention (squeeze-and-excitation style)
/// for 2D tensors shaped [batch, features].
#[derive(Debug)]
pub struct Attention {
    w1: nn::Linear,
    w2: nn::Linear,
}

impl Attention {
    /// Create a new Attention module for feature dimension `dim`.
    /// `reduction` controls the hidden size = max(1, dim / reduction).
    pub fn new(p: &nn::Path, dim: i64, reduction: i64) -> Self {
        let hidden = (dim / reduction).max(1);
        let w1 = nn::linear(p / "w1", dim, hidden, Default::default());
        let w2 = nn::linear(p / "w2", hidden, dim, Default::default());
        Self { w1, w2 }
    }

    /// Forward: gate = sigmoid(W2(relu(W1(x)))) ; return x * gate
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = x.apply(&self.w1).relu().apply(&self.w2).sigmoid();
        x * gate
    }
}
