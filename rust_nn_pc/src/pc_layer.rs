use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct PCLayer {
    // Latent state x, optimized during inference
    x: Option<Tensor>,
    // Sum energy (scalar) from last forward in train mode
    energy: Option<Tensor>,
    // Whether to sample/initialize x at next forward
    is_sample_x: bool,
}

impl PCLayer {
    pub fn new() -> Self {
        Self {
            x: None,
            energy: None,
            is_sample_x: false,
        }
    }

    pub fn set_sample_x(&mut self, v: bool) {
        self.is_sample_x = v;
    }
    pub fn energy(&self) -> Option<Tensor> {
        self.energy.as_ref().map(|t| t.shallow_clone())
    }
    pub fn x(&self) -> Option<Tensor> {
        self.x.as_ref().map(|t| t.shallow_clone())
    }

    fn needs_sampling(&self, mu: &Tensor) -> bool {
        if self.is_sample_x {
            return true;
        }
        match &self.x {
            None => true,
            Some(x) => {
                let same_device = x.device() == mu.device();
                let same_size = x.size() == mu.size();
                !(same_device && same_size)
            }
        }
    }

    fn sample_x_from_mu(&mut self, mu: &Tensor) -> Tensor {
        // Default: x = mu (detached) so it's a leaf, then enable grad
        let new_x = mu.detach().set_requires_grad(true);
        self.x = Some(new_x.shallow_clone());
        self.is_sample_x = false; // one-shot
        new_x
    }

    // Forward in training: hold energy and return x
    pub fn forward_train(&mut self, mu: &Tensor) -> Tensor {
        let x = if self.needs_sampling(mu) {
            self.sample_x_from_mu(mu)
        } else {
            // If somehow missing, sample now; avoid unwrap/expect
            self.x
                .as_ref()
                .map(|t| t.shallow_clone())
                .unwrap_or_else(|| self.sample_x_from_mu(mu))
        };

        // Energy: 0.5 * (mu - x)^2 summed over all but batch dims as in python default
        // Here we compute a scalar energy for the whole batch (sum of all elements)
        let diff = mu - &x;
        let energy = ((&diff * &diff).sum(Kind::Float)) * 0.5;
        self.energy = Some(energy);

        x.shallow_clone()
    }

    // Forward in eval: transparent (no energy accumulation)
    pub fn forward_eval(&self, mu: &Tensor) -> Tensor {
        mu.shallow_clone()
    }

    // Perform one gradient descent step on x with given learning rate
    pub fn step_x(&mut self, lr: f64) {
        if let Some(x) = &self.x {
            let grad = x.grad();
            if grad.defined() {
                let new_x = x - &grad * lr;
                self.x = Some(new_x.detach().set_requires_grad(true));
            }
        }
    }
}

impl Default for PCLayer {
    fn default() -> Self {
        Self::new()
    }
}
