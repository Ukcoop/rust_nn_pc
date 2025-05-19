use std::error::Error;

pub mod matrix;
use matrix::Matrix;

fn sigmoid_vec(x: &Matrix) -> Matrix {
    let mut result = x.clone();
    result.map_inplace(|v| 1.0 / (1.0 + (-v).exp()));
    result
}

fn sigmoid_derivative(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let mut one = Matrix::new(a.rows, a.cols, a.parallelize_threshold);
    one.add_scalar(1.0);

    let tmp = Matrix::subtract(&one, a)?;
    let result = Matrix::elementwise_multiply(a, &tmp)?;

    return Ok(result);
}

#[derive(Debug, Default)]
pub struct DenseLayer {
    pub weights: Matrix,
    pub bias: Matrix,
    pub activation: Matrix,
    pub z: Matrix,
    pub delta: Matrix,
}

impl DenseLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        parallelize_threshold: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let mut w = Matrix::new(output_dim, input_dim, parallelize_threshold);
        w.randomize();
        Ok(DenseLayer {
            weights: w,
            bias: Matrix::new(output_dim, 1, parallelize_threshold),
            activation: Matrix::new(output_dim, 1, parallelize_threshold),
            z: Matrix::new(output_dim, 1, parallelize_threshold),
            delta: Matrix::new(output_dim, 1, parallelize_threshold),
        })
    }

    pub fn forward(&mut self, input: &Matrix) -> Result<(), Box<dyn Error>> {
        self.z = Matrix::multiply(&self.weights, input)?;
        self.z.add(&self.bias)?;
        self.activation = sigmoid_vec(&self.z);

        return Ok(());
    }

    pub fn compute_delta(
        &mut self,
        downstream_w: Option<&Matrix>,
        downstream_delta: Option<&Matrix>,
        target: Option<&Matrix>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(t) = target {
            let err = Matrix::subtract(&self.activation, t)?;
            let deriv = sigmoid_derivative(&self.activation)?;
            self.delta = Matrix::elementwise_multiply(&err, &deriv)?;
        } else if let (Some(w), Some(d)) = (downstream_w, downstream_delta) {
            let wt = Matrix::transpose(w);
            let temp = Matrix::multiply(&wt, d)?;
            let deriv = sigmoid_derivative(&self.activation)?;
            self.delta = Matrix::elementwise_multiply(&temp, &deriv)?;
        }

        return Ok(());
    }

    pub fn update(&mut self, input: &Matrix) -> Result<(), Box<dyn Error>> {
        let input_t = Matrix::transpose(input);
        let grad_w = Matrix::multiply(&self.delta, &input_t)?;
        self.weights = Matrix::subtract(&self.weights, &grad_w)?;
        self.bias = Matrix::subtract(&self.bias, &self.delta)?;

        return Ok(());
    }
}

pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
    pub parallelize_threshold: usize,
}

impl NeuralNetwork {
    pub fn new(sizes: &[usize]) -> Result<Self, Box<dyn Error>> {
        let parallelize_threshold: usize = rayon::current_num_threads() * 10;

        let layers = sizes
            .windows(2)
            .map(|w| DenseLayer::new(w[0], w[1], parallelize_threshold).unwrap_or_default())
            .collect();
        Ok(NeuralNetwork {
            layers,
            parallelize_threshold,
        })
    }

    pub fn forward(&mut self, input: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        let mut activation = input.clone();
        for layer in &mut self.layers {
            layer.forward(&activation)?;
            activation = layer.activation.clone();
        }

        return Ok(activation);
    }

    pub fn backward(&mut self, input: &Matrix, target: &Matrix) -> Result<(), Box<dyn Error>> {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.clone());
        for layer in &mut self.layers {
            if let Some(last) = activations.last() {
                layer.forward(last)?;
                activations.push(layer.activation.clone());
            }
        }
        let mut next_w: Option<&Matrix> = None;
        let mut next_delta: Option<&Matrix> = None;

        let layers_len = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            if i == layers_len - 1 {
                layer.compute_delta(None, None, Some(target))?;
            } else {
                layer.compute_delta(next_w, next_delta, None)?;
            }
            next_w = Some(&layer.weights);
            next_delta = Some(&layer.delta);
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.update(&activations[i])?;
        }

        return Ok(());
    }

    pub fn train(
        &mut self,
        input: &[f32],
        target: &[f32],
        epochs: usize,
    ) -> Result<(), Box<dyn Error>> {
        let inp = Matrix::from_vector(input, self.parallelize_threshold);
        let tgt = Matrix::from_vector(target, self.parallelize_threshold);
        for _ in 0..epochs {
            self.backward(&inp, &tgt)?;
        }

        return Ok(());
    }

    pub fn predict(&mut self, input: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let out = self.forward(&Matrix::from_vector(input, self.parallelize_threshold))?;
        return Ok(out.to_vector());
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
                nn.train(input, target, 1).unwrap();
            }

            let mut total_error = 0.0;
            for (input, target) in &inputs {
                let output = nn.predict(input).unwrap();
                total_error += (target[0] - output[0]).abs();
            }

            if total_error < 0.005 {
                break;
            }
        }
    }
}
