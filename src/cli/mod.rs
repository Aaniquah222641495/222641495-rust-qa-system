// Layer 1 — CLI / Presentation Layer
pub mod commands;

use anyhow::Result;
use clap::Parser;
use commands::{Commands, TrainArgs, AskArgs};

#[derive(Parser, Debug)]
#[command(name = "word-doc-qa", version = "0.1.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

impl Cli {
    pub fn run(self) -> Result<()> {
        match self.command {
            Commands::Train(args) => run_train(args),
            Commands::Ask(args)   => run_ask(args),
        }
    }
}

fn run_train(args: TrainArgs) -> Result<()> {
    use crate::application::train_use_case::{TrainUseCase, TrainConfig};
    tracing::info!("Starting training on documents in: {}", args.docs_dir);

    let cfg = TrainConfig {
        vocab_size:     args.vocab_size,
        max_seq_len:    args.max_seq_len,
        d_model:        args.d_model,
        num_heads:      args.num_heads,
        num_layers:     args.num_layers,
        d_ff:           args.d_ff,
        dropout:        args.dropout,
        batch_size:     args.batch_size,
        epochs:         args.epochs,
        lr:             args.lr,
        docs_dir:       args.docs_dir,
        checkpoint_dir: args.checkpoint_dir,
    };

    let use_case = TrainUseCase::new(cfg);
    use_case.execute()?;
    println!("Training complete. Checkpoint saved.");
    Ok(())
}

fn run_ask(args: AskArgs) -> Result<()> {
    use crate::application::ask_use_case::AskUseCase;
    let use_case = AskUseCase::new(
        args.checkpoint_dir.clone(),
        args.docs_dir.clone()
    )?;
    let answer = use_case.answer(&args.question)?;
    println!("\nAnswer: {}", answer);
    Ok(())
}
