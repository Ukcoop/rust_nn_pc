use std::error::Error;

use ndarray::{Array1, Array2};
use rand::random_range;

fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn sigmoid_derivative(a: &Array1<f32>) -> Array1<f32> {
    a * &(1.0 - a)
}

#[derive(Default)]
pub struct DenseLayer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: Array1<f32>,
    pub z: Array1<f32>,
    pub delta: Array1<f32>,
}

impl DenseLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self, Box<dyn Error>> {
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|_| random_range(-1.0..1.0))
            .collect();

        let weights = Array2::from_shape_vec((output_dim, input_dim), weight_data)?;

        return Ok(DenseLayer {
            weights,
            bias: Array1::zeros(output_dim),
            activation: Array1::zeros(output_dim),
            z: Array1::zeros(output_dim),
            delta: Array1::zeros(output_dim),
        });
    }

    pub fn forward(&mut self, input: &Array1<f32>) {
        self.z = self.weights.dot(input) + &self.bias;
        self.activation = sigmoid(&self.z);
    }

    pub fn compute_delta(
        &mut self,
        downstream_w: Option<&Array2<f32>>,
        downstream_delta: Option<&Array1<f32>>,
        target: Option<&Array1<f32>>,
    ) {
        if let Some(t) = target {
            let error = &self.activation - t;
            self.delta = error * &sigmoid_derivative(&self.activation);
        } else if let (Some(w), Some(d)) = (downstream_w, downstream_delta) {
            self.delta = w.t().dot(d) * &sigmoid_derivative(&self.activation);
        }
    }

    pub fn update(&mut self, input: &Array1<f32>) {
        let grad_w = self
            .delta
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&input.view().insert_axis(ndarray::Axis(0)));
        self.weights.zip_mut_with(&grad_w, |w, &g| *w -= g);
        self.bias.zip_mut_with(&self.delta, |b, &d| *b -= d);
    }
}

pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    pub fn new(sizes: &[usize]) -> Result<Self, Box<dyn Error>> {
        let layers = sizes
            .windows(2)
            .map(|win| DenseLayer::new(win[0], win[1]).unwrap_or_default())
            .collect();

        return Ok(NeuralNetwork { layers });
    }

    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut activation = input.clone();
        for layer in self.layers.iter_mut() {
            layer.forward(&activation);
            activation = layer.activation.clone();
        }

        return activation;
    }

    pub fn backward(&mut self, input: &Array1<f32>, target: &Array1<f32>) {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.clone());
        for layer in self.layers.iter_mut() {
            if let Some(last) = activations.last() {
                layer.forward(last);
                activations.push(layer.activation.clone());
            }
        }

        let mut next_w: Option<&Array2<f32>> = None;
        let mut next_delta: Option<&Array1<f32>> = None;
        let layers_len = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            if i == layers_len - 1 {
                layer.compute_delta(None, None, Some(target));
            } else {
                layer.compute_delta(next_w, next_delta, None);
            }
            next_w = Some(&layer.weights);
            next_delta = Some(&layer.delta);
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.update(&activations[i]);
        }
    }

    pub fn train(&mut self, input: &[f32], target: &[f32], epochs: usize) {
        for _ in 0..epochs {
            self.backward(
                &Array1::from_vec(input.to_owned()),
                &Array1::from_vec(target.to_owned()),
            );
        }
    }

    pub fn predict(&mut self, input: &[f32]) -> Array1<f32> {
        self.forward(&Array1::from_vec(input.to_owned()))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn xor_problem() {
        let mut nn = NeuralNetwork::new(&[2, 4, 1]).unwrap();

        let inputs = vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        loop {
            for (input, target) in &inputs {
                nn.train(input, target, 1);
            }

            let mut total_error = 0.0;
            for (input, target) in &inputs {
                let output = nn.predict(input);
                total_error += (target[0] - output[0]).abs();
            }

            if total_error < 0.005 {
                break;
            }
        }
    }
}
