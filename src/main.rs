#![allow(dead_code, unused_imports)]
#![recursion_limit = "256"]

mod cli;
mod application;
mod domain;
mod data;
mod ml;
mod infra;

use anyhow::Result;
use cli::Cli;
use clap::Parser;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("word_doc_qa=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();
    cli.run()
}
