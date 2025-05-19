use rust_nn_pc::NeuralNetwork;

use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let mut nn = NeuralNetwork::new(&[2, 4, 4, 1])?;

    let inputs = vec![
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let total_timer = Instant::now();
    let mut cycle_count: usize = 0;

    loop {
        for (input, target) in &inputs {
            nn.train(input, target, 1)?;
            cycle_count += 1;
        }

        let mut total_error = 0.0;
        for (input, target) in &inputs {
            let output = nn.predict(input)?;
            total_error += (target[0] - output[0]).abs();
        }

        if total_error < 0.005 {
            let total_duration = total_timer.elapsed();
            let seconds = total_duration.as_secs_f64();
            let cycles_per_sec = cycle_count as f64 / seconds;

            println!("Training complete!");
            println!("Final error: {:.6}", total_error);
            println!("Cycles per second: {:.2}", cycles_per_sec);
            println!("Total time: {:.2?}", total_duration);
            break;
        }
    }

    Ok(())
}
