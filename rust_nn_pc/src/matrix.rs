use nalgebra::DMatrix;
use rayon::prelude::*;
use std::error::Error;

#[derive(Debug, Clone, Default)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: DMatrix<f32>,
    pub parallelize_threshold: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, parallelize_threshold: usize) -> Matrix {
        return Matrix {
            rows,
            cols,
            data: DMatrix::zeros(rows, cols),
            parallelize_threshold,
        };
    }

    pub fn from_vector(vector: &[f32], parallelize_threshold: usize) -> Matrix {
        let data = DMatrix::from_vec(vector.len(), 1, vector.to_owned());

        return Matrix {
            rows: vector.len(),
            cols: 1,
            data,
            parallelize_threshold,
        };
    }

    pub fn to_vector(&self) -> Vec<f32> {
        return self.data.as_slice().to_vec();
    }

    pub fn transpose(&self) -> Matrix {
        let data = self.data.transpose();

        return Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
            parallelize_threshold: self.parallelize_threshold,
        };
    }

    pub fn randomize(&mut self) {
        let slice = self.data.as_mut_slice();

        if slice.len() >= self.parallelize_threshold {
            slice.par_iter_mut().for_each(|x| *x = rand::random());
        } else {
            slice.iter_mut().for_each(|x| *x = rand::random());
        }
    }

    pub fn add(&mut self, other: &Matrix) -> Result<(), Box<dyn Error>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix shapes must match for add".into());
        }

        let slice = self.data.as_mut_slice();
        let other_slice = other.data.as_slice();

        if slice.len() >= self.parallelize_threshold {
            slice
                .par_iter_mut()
                .zip(other_slice.par_iter())
                .for_each(|(a, &b)| *a += b);
        } else {
            slice
                .iter_mut()
                .zip(other_slice.iter())
                .for_each(|(a, &b)| *a += b);
        }

        return Ok(());
    }

    pub fn add_scalar(&mut self, number: f32) {
        let slice = self.data.as_mut_slice();

        if slice.len() >= self.parallelize_threshold {
            slice.par_iter_mut().for_each(|x| *x += number);
        } else {
            slice.iter_mut().for_each(|x| *x += number);
        }
    }

    pub fn subtract(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err("Matrix shapes must match for subtract".into());
        }

        let mut result = Matrix::new(a.rows, a.cols, b.parallelize_threshold);
        let len = a.data.as_slice().len();

        if len >= a.parallelize_threshold {
            let cols = a.cols;

            a.data
                .as_slice()
                .par_chunks(cols)
                .zip(b.data.as_slice().par_chunks(cols))
                .zip(result.data.as_mut_slice().par_chunks_mut(cols))
                .for_each(|((ar, br), rr)| {
                    for i in 0..cols {
                        rr[i] = ar[i] - br[i];
                    }
                });
        } else {
            result.data = &a.data - &b.data;
        }

        return Ok(result);
    }

    pub fn elementwise_multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err("Matrix shapes must match for elementwise multiply".into());
        }

        let mut result = Matrix::new(a.rows, a.cols, a.parallelize_threshold);
        let len = a.data.as_slice().len();

        let aslice = a.data.as_slice();
        let bslice = b.data.as_slice();
        let rslice = result.data.as_mut_slice();

        if len >= a.parallelize_threshold {
            rslice.par_iter_mut().enumerate().for_each(|(i, slot)| {
                *slot = aslice[i] * bslice[i];
            });
        } else {
            for i in 0..len {
                rslice[i] = aslice[i] * bslice[i];
            }
        }

        return Ok(result);
    }

    pub fn multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.cols != b.rows {
            return Err("Matrix shapes not conformable for multiply".into());
        }

        let mut result = Matrix::new(a.rows, b.cols, b.parallelize_threshold);
        let len = a.data.as_slice().len();

        if len >= a.parallelize_threshold {
            let cols = b.cols;
            let inner = a.cols;

            result
                .data
                .as_mut_slice()
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(i, row_out)| {
                    for j in 0..cols {
                        let mut sum = 0.0;
                        for k in 0..inner {
                            sum += a.data[(i, k)] * b.data[(k, j)];
                        }
                        row_out[j] = sum;
                    }
                });
        } else {
            result.data = &a.data * &b.data;
        }

        return Ok(result);
    }

    pub fn scale(&mut self, scalar: f32) {
        let slice = self.data.as_mut_slice();

        if slice.len() >= self.parallelize_threshold {
            slice.par_iter_mut().for_each(|x| *x *= scalar);
        } else {
            slice.iter_mut().for_each(|x| *x *= scalar);
        }
    }

    pub fn map_inplace<F>(&mut self, func: F)
    where
        F: Fn(f32) -> f32 + Sync + Send,
    {
        let slice = self.data.as_mut_slice();

        if slice.len() >= self.parallelize_threshold {
            slice.par_iter_mut().for_each(|x| *x = func(*x));
        } else {
            slice.iter_mut().for_each(|x| *x = func(*x));
        }
    }

    pub fn static_map<F>(matrix: &Matrix, func: F) -> Matrix
    where
        F: Fn(f32) -> f32 + Sync + Send + Clone,
    {
        let mut result = Matrix::new(matrix.rows, matrix.cols, matrix.parallelize_threshold);

        let len = matrix.data.as_slice().len();
        let src = matrix.data.as_slice();
        let dst = result.data.as_mut_slice();

        if len >= matrix.parallelize_threshold {
            dst.par_iter_mut().enumerate().for_each(|(i, slot)| {
                *slot = func(src[i]);
            });
        } else {
            for i in 0..len {
                dst[i] = func(src[i]);
            }
        }

        return result;
    }
}
