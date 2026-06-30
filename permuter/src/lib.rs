#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::cast_precision_loss,
    clippy::while_float,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc
)]

use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const BLOCK_SIZE: usize = 4;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(transparent)]
pub struct Matrix {
    pub data: Vec<Vec<u32>>,
}

#[must_use]
pub fn cost_function(order: &[u16], matrix: &Matrix) -> u64 {
    (BLOCK_SIZE..order.len() - BLOCK_SIZE)
        .into_par_iter()
        .map(|i| {
            let matrix_row = &matrix.data[order[i] as usize][..];
            let cost_a = order[i + BLOCK_SIZE..]
                .iter()
                .map(|&j| u64::from(matrix_row[j as usize]))
                .sum::<u64>();
            let cost_b = order[..i - BLOCK_SIZE]
                .iter()
                .map(|&j| u64::from(matrix_row[j as usize]))
                .sum::<u64>();
            cost_a + cost_b
        })
        .sum()
}

fn row_sums(matrix: &Matrix) -> Vec<u64> {
    matrix
        .data
        .iter()
        .map(|row| row.iter().map(|&v| u64::from(v)).sum())
        .collect()
}

#[derive(Clone, Copy)]
enum Move {
    Swap(usize, usize),
    Reverse(usize, usize),
}

impl Move {
    fn apply(self, order: &mut [u16]) {
        match self {
            Self::Swap(a, b) => order.swap(a, b),
            Self::Reverse(s, e) => order[s..=e].reverse(),
        }
    }
}

struct Evaluator<'a> {
    matrix: &'a Matrix,
    row_sums: Vec<u64>,
    recompute: bool,
}

impl<'a> Evaluator<'a> {
    fn new(matrix: &'a Matrix, recompute: bool) -> Self {
        Self {
            row_sums: row_sums(matrix),
            matrix,
            recompute,
        }
    }

    fn cost(&self, order: &[u16]) -> u64 {
        cost_function(order, self.matrix)
    }

    #[inline]
    fn near_band(&self, order: &[u16], i: usize) -> u64 {
        let row = &self.matrix.data[order[i] as usize];
        order[(i - BLOCK_SIZE)..(i + BLOCK_SIZE)]
            .iter()
            .map(|&k| u64::from(row[k as usize]))
            .sum()
    }

    #[inline]
    fn partial_cost(&self, order: &[u16], lo: usize, hi: usize) -> i64 {
        let hi = hi.min(order.len() - BLOCK_SIZE - 1);
        (lo.max(BLOCK_SIZE)..=hi)
            .map(|i| self.row_sums[order[i] as usize] as i64 - self.near_band(order, i) as i64)
            .sum()
    }

    fn swap_windows_cost(&self, order: &[u16], a: usize, b: usize) -> i64 {
        let (p, q) = (a.min(b), a.max(b));
        let (lo_p, lo_q) = (
            p.saturating_sub(BLOCK_SIZE - 1),
            q.saturating_sub(BLOCK_SIZE - 1),
        );
        if q < p + 2 * BLOCK_SIZE {
            self.partial_cost(order, lo_p, q + BLOCK_SIZE)
        } else {
            self.partial_cost(order, lo_p, p + BLOCK_SIZE)
                + self.partial_cost(order, lo_q, q + BLOCK_SIZE)
        }
    }

    fn affected_cost(&self, order: &[u16], mv: Move) -> i64 {
        match mv {
            Move::Swap(a, b) => self.swap_windows_cost(order, a, b),
            Move::Reverse(s, e) => {
                self.partial_cost(order, s.saturating_sub(BLOCK_SIZE - 1), e + BLOCK_SIZE)
            }
        }
    }

    fn apply_move(&self, order: &mut [u16], mv: Move) -> i64 {
        if self.recompute {
            let before = self.cost(order) as i64;
            mv.apply(order);
            self.cost(order) as i64 - before
        } else {
            let before = self.affected_cost(order, mv);
            mv.apply(order);
            self.affected_cost(order, mv) - before
        }
    }

    fn undo_move(&self, order: &mut [u16], mv: Move) {
        mv.apply(order);
    }
}

fn distinct_pair(rng: &mut impl Rng, len: usize) -> (usize, usize) {
    let a = rng.random_range(0..len);
    let mut b = rng.random_range(0..len);
    while b == a {
        b = rng.random_range(0..len);
    }
    (a, b)
}

fn select_move(rng: &mut impl Rng, len: usize) -> Move {
    match rng.random_range(0..3) {
        0 => {
            let i = rng.random_range(0..len - 1);
            Move::Swap(i, i + 1)
        }
        1 => {
            let (i, j) = distinct_pair(rng, len);
            Move::Swap(i, j)
        }
        _ => {
            let (mut start, mut end) = distinct_pair(rng, len);
            if start > end {
                (start, end) = (end, start);
            }
            Move::Reverse(start, end)
        }
    }
}

pub fn greedy_sort(order: &mut [u16], matrix: &Matrix) {
    let start = Instant::now();
    let eval = Evaluator::new(matrix, false);
    let mut cost = eval.cost(order);
    let mut improvements = 0;
    let mut iters = 0u64;
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..order.len() - 1 {
            for j in i + 1..order.len() {
                iters += 1;
                let mv = Move::Swap(i, j);
                let delta = eval.apply_move(order, mv);
                if delta >= 0 {
                    eval.undo_move(order, mv);
                    continue; // Skip if swap does not improve cost
                }
                cost = cost.wrapping_add_signed(delta);
                changed = true;
                improvements += 1;
            }
        }
    }
    println!(
        "Greedy sort completed in {:.2}s with {improvements} improvements and {iters} iterations. Final cost: {cost}",
        start.elapsed().as_secs_f64(),
    );
}

pub fn simulated_annealing(
    order: &mut [u16],
    matrix: &Matrix,
    rng: &mut impl Rng,
    initial_temp: f64,
    cooling_rate: f64,
    min_temp: f64,
    recompute: bool,
) {
    let start = Instant::now();
    let eval = Evaluator::new(matrix, recompute);
    let mut current_cost = eval.cost(order);
    let mut best_cost = current_cost;
    let mut best_order = order.to_vec();
    let mut temp = initial_temp;
    let mut iterations = 0u64;
    let mut improvements = 0;

    // Calculate total iterations: initial_temp * cooling_rate^n = min_temp
    // => n = log(min_temp / initial_temp) / log(cooling_rate)
    let total_iterations = (min_temp / initial_temp).log(cooling_rate).ceil() as u64;

    let pb = ProgressBar::new(total_iterations);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (T={msg}) ETA: {eta}")
            .expect("valid template")
            .progress_chars("#>-"),
    );

    while temp > min_temp {
        let mv = select_move(rng, order.len());
        let delta = eval.apply_move(order, mv);
        let new_cost = current_cost.wrapping_add_signed(delta);

        // make sure incremental eval didn't drift
        debug_assert_eq!(new_cost, eval.cost(order));

        // if energy has gone up, then exp(-delta_energy / temp) is greater than 1
        // otherwise, it is some number between 0 and 1. We always accept edits
        // that lower the energy, and accept some that raise it based on the temperature.
        // This is the essence of simulated annealing.
        if delta < 0 || (-delta as f64 / temp).exp() > rng.random::<f64>() {
            // Accept the new order
            current_cost = new_cost;
            if current_cost < best_cost {
                best_cost = current_cost;
                best_order.copy_from_slice(order);
                improvements += 1;
            }
        } else {
            eval.undo_move(order, mv);
        }

        temp *= cooling_rate;
        iterations += 1;

        if iterations.is_multiple_of(1_000) {
            pb.set_position(iterations);
            pb.set_message(format!("{temp:.2e}"));
        }
    }

    pb.finish_and_clear();
    order.copy_from_slice(&best_order);
    println!(
        "Simulated annealing completed in {:.2}s with {improvements} improvements and {iterations} iterations. Final cost: {best_cost}",
        start.elapsed().as_secs_f64()
    );
}
