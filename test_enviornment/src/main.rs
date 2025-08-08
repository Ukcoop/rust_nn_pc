use rust_nn_pc::NeuralNetwork;
use tch::{Device, vision};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    // Data: MNIST flattened
    let m = vision::mnist::load_dir("./data/MNIST/raw")?;
    let train_images = m.train_images.to_device(device).view([-1, 28 * 28]);
    let train_labels = m.train_labels.to_device(device);
    let test_images = m.test_images.to_device(device).view([-1, 28 * 28]);
    let test_labels = m.test_labels.to_device(device);

    // Simple API usage
    let mut nn = NeuralNetwork::new([28 * 28, 256, 256, 10])?;
    nn.set_t_steps(20);
    nn.set_lr_x(1e-2);

    let batch_size: i64 = 500;
    loop {
        let num_train = train_images.size()[0];
        for start in (0..num_train).step_by(batch_size as usize) {
            let end = (start + batch_size).min(num_train);
            // build a batch and call train_batch
            let mut xs: Vec<Vec<f32>> = Vec::new();
            let mut ys: Vec<Vec<f32>> = Vec::new();
            for i in start..end {
                let img: Vec<f32> = train_images.get(i).to_device(Device::Cpu).try_into()?;
                let lbl = train_labels.get(i).int64_value(&[]);
                let mut one_hot = vec![0.0f32; 10];
                one_hot[lbl as usize] = 1.0;
                xs.push(img);
                ys.push(one_hot);
            }
            nn.train_batch(xs, ys)?;
        }

        // Evaluate: print Mean Squared Error (MSE) against one-hot labels
        let mut se_sum = 0.0f64; // sum of squared errors over all samples and classes
        let mut count = 0usize; // total number of scalar predictions
        for i in 0..test_images.size()[0] {
            let img: Vec<f32> = test_images.get(i).to_device(Device::Cpu).try_into()?;
            let pred = nn.predict(img)?;
            let lbl = test_labels.get(i).int64_value(&[]) as usize;
            for (j, &p) in pred.iter().enumerate() {
                let target = if j == lbl { 1.0f32 } else { 0.0f32 };
                let diff = (p as f64) - (target as f64);
                se_sum += diff * diff;
                count += 1;
            }
        }
        let mse = se_sum / (count as f64);
        println!("Error: {mse:.6}");
        if mse < 0.005 {
            // Save when threshold reached
            nn.save("./model.ot")?;
            break;
        }
    }

    Ok(())
}
