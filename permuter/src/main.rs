#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_possible_truncation)]

use std::{cmp::Reverse, time::Instant};

use anyhow::{Context, ensure};
use permuter::{Matrix, cost_function, greedy_sort, simulated_annealing};
use rand::SeedableRng;

fn main() -> anyhow::Result<()> {
    let start = Instant::now();

    let matrix =
        std::fs::read_to_string("correlations.json").context("Failed to read correlations.json")?;
    let matrix =
        serde_json::from_str::<Matrix>(&matrix).context("Failed to parse correlations.json")?;

    ensure!(
        matrix.data.iter().all(|row| row.len() == matrix.data.len()),
        "Matrix must be square"
    );

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let diag = matrix
        .data
        .iter()
        .enumerate()
        .map(|(i, row)| row[i])
        .collect::<Vec<_>>();

    let default_order = (0..diag.len() as u16).collect::<Vec<_>>();

    let mut sorted_indices = default_order.clone();
    sorted_indices.sort_unstable_by_key(|&i| Reverse(diag[i as usize]));

    std::fs::write("sorted_order.json", serde_json::to_string(&sorted_indices)?)
        .context("Failed to write sorted_order.json")?;

    println!(
        "Cost of default order: {}",
        cost_function(&default_order, &matrix)
    );
    println!(
        "Cost of sorted order: {}",
        cost_function(&sorted_indices, &matrix)
    );

    let recompute = std::env::args().any(|arg| arg == "--recompute");

    let mut annealed_order = sorted_indices.clone();
    simulated_annealing(
        &mut annealed_order,
        &matrix,
        &mut rng,
        1000.0,   // Initial temperature
        0.999999, // Cooling rate
        1e-6,     // Minimum temperature
        recompute,
    );

    greedy_sort(&mut annealed_order, &matrix);

    std::fs::write("final_order.json", serde_json::to_string(&annealed_order)?)
        .context("Failed to write final_order.json")?;

    println!("Total time: {:.2}s", start.elapsed().as_secs_f64());

    Ok(())
}
