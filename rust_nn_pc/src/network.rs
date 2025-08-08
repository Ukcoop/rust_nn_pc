use crate::pc_layer::PCLayer;
use tch::{Device, Kind, Result, Tensor, nn, nn::OptimizerConfig};

pub struct NeuralNetwork {
    device: Device,
    vs: nn::VarStore,
    linears: Vec<nn::Linear>,
    pc_layers: Vec<PCLayer>, // one per hidden layer
    opt: nn::Optimizer,
    // PC hyperparameters
    t_steps: usize,
    lr_x: f64,
}

impl NeuralNetwork {
    pub fn new<const N: usize>(dims: [i64; N]) -> Result<Self> {
        assert!(dims.len() >= 2, "need at least input and output dims");
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let vs = nn::VarStore::new(device);
        let root = &vs.root();

        // Build linears
        let mut linears = Vec::new();
        for w in dims.as_slice().windows(2) {
            let linear = nn::linear(
                root / format!("lin_{}{}_{}", w[0], w[1], linears.len()),
                w[0],
                w[1],
                Default::default(),
            );
            linears.push(linear);
        }

        // PCLayers for each hidden transformation (i.e., after all but last linear)
        let mut pc_layers = Vec::new();
        for _ in 0..(linears.len().saturating_sub(1)) {
            pc_layers.push(PCLayer::new());
        }

        let opt = nn::Adam::default().build(&vs, 1e-3)?;

        Ok(Self {
            device,
            vs,
            linears,
            pc_layers,
            opt,
            t_steps: 20,
            lr_x: 1e-2,
        })
    }

    pub fn set_t_steps(&mut self, t: usize) {
        self.t_steps = t;
    }
    pub fn set_lr_x(&mut self, lr_x: f64) {
        self.lr_x = lr_x;
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.vs.save(path)
    }

    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        self.vs.load(path)
    }

    fn forward_eval_internal(&self, input: &Tensor) -> Tensor {
        let mut x = input.shallow_clone();
        for i in 0..self.linears.len() {
            x = x.apply(&self.linears[i]);
            if i + 1 != self.linears.len() {
                x = x.relu();
            }
        }
        x
    }

    fn forward_pc_train_internal(&mut self, input: &Tensor) -> Tensor {
        let mut x = input.apply(&self.linears[0]);
        // hidden layers with PC
        for i in 0..self.pc_layers.len() {
            x = self.pc_layers[i].forward_train(&x).relu();
            x = x.apply(&self.linears[i + 1]);
        }
        // x now holds the pre-activation logits at output (no pc layer)
        x
    }

    pub fn predict(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        let in_dim = self.linears[0].ws.size()[1];
        assert_eq!(input.len() as i64, in_dim, "input dim mismatch");
        let x = Tensor::f_from_slice(&input)?
            .to_device(self.device)
            .view([1, in_dim]);
        let y = self.forward_eval_internal(&x);
        let v: Vec<f32> = y.squeeze_dim(0).to_device(Device::Cpu).try_into()?;
        Ok(v)
    }

    pub fn train(&mut self, input: Vec<f32>, target: Vec<f32>) -> Result<()> {
        let in_dim = self.linears[0].ws.size()[1];
        let out_dim = self.linears[self.linears.len() - 1].ws.size()[0];
        assert_eq!(input.len() as i64, in_dim, "input dim mismatch");
        assert_eq!(target.len() as i64, out_dim, "target dim mismatch");

        let x = Tensor::f_from_slice(&input)?
            .to_device(self.device)
            .view([1, in_dim]);
        let y_t = Tensor::f_from_slice(&target)?
            .to_device(self.device)
            .view([1, out_dim]);

        // Sample x at batch start
        for pc in &mut self.pc_layers {
            pc.set_sample_x(true);
        }

        // Inference loop: update xs only
        for _ in 0..self.t_steps {
            let logits = self.forward_pc_train_internal(&x);
            // loss: 0.5 * (logits - y_t)^2
            let diff = &logits - &y_t;
            let loss = (&diff * &diff).sum(Kind::Float) * 0.5;
            let mut overall = loss;
            for pc in &self.pc_layers {
                if let Some(e) = pc.energy() {
                    overall += e;
                }
            }
            overall.backward();
            for pc in &mut self.pc_layers {
                pc.step_x(self.lr_x);
            }
            tch::no_grad(|| {
                for (_, mut t) in self.vs.variables() {
                    t.zero_grad();
                }
            });
        }

        // Parameter update
        self.opt.zero_grad();
        let logits = self.forward_pc_train_internal(&x);
        let diff = &logits - &y_t;
        let loss = (&diff * &diff).sum(Kind::Float) * 0.5;
        let mut overall = loss;
        for pc in &self.pc_layers {
            if let Some(e) = pc.energy() {
                overall += e;
            }
        }
        overall.backward();
        self.opt.step();
        Ok(())
    }

    // Batched variant for speed: each inner Vec<f32> is one sample
    pub fn train_batch(&mut self, inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Result<()> {
        assert_eq!(inputs.len(), targets.len(), "batch size mismatch");
        let batch = inputs.len() as i64;
        let in_dim = self.linears[0].ws.size()[1];
        let out_dim = self.linears[self.linears.len() - 1].ws.size()[0];
        for inp in &inputs {
            assert_eq!(inp.len() as i64, in_dim, "input dim mismatch");
        }
        for tgt in &targets {
            assert_eq!(tgt.len() as i64, out_dim, "target dim mismatch");
        }

        // Flatten and create tensors [B, Din], [B, Dout]
        let flat_in: Vec<f32> = inputs.into_iter().flatten().collect();
        let flat_tg: Vec<f32> = targets.into_iter().flatten().collect();
        let x = Tensor::f_from_slice(&flat_in)?
            .to_device(self.device)
            .view([batch, in_dim]);
        let y_t = Tensor::f_from_slice(&flat_tg)?
            .to_device(self.device)
            .view([batch, out_dim]);

        // Sample x at batch start
        for pc in &mut self.pc_layers {
            pc.set_sample_x(true);
        }

        // Inference loop: update xs only
        for _ in 0..self.t_steps {
            let logits = self.forward_pc_train_internal(&x);
            let diff = &logits - &y_t;
            let loss = (&diff * &diff).sum(Kind::Float) * 0.5;
            let mut overall = loss;
            for pc in &self.pc_layers {
                if let Some(e) = pc.energy() {
                    overall += e;
                }
            }
            overall.backward();
            for pc in &mut self.pc_layers {
                pc.step_x(self.lr_x);
            }
            tch::no_grad(|| {
                for (_, mut t) in self.vs.variables() {
                    t.zero_grad();
                }
            });
        }

        // Parameter update
        self.opt.zero_grad();
        let logits = self.forward_pc_train_internal(&x);
        let diff = &logits - &y_t;
        let loss = (&diff * &diff).sum(Kind::Float) * 0.5;
        let mut overall = loss;
        for pc in &self.pc_layers {
            if let Some(e) = pc.energy() {
                overall += e;
            }
        }
        overall.backward();
        self.opt.step();
        Ok(())
    }
}
