// ============================================================
// Layer 1 — CLI / Presentation Layer
// ============================================================
// This is the entry point for all user interaction.
// It uses the `clap` crate to parse command line arguments.
// All business logic is delegated to Layer 2 (application).
//
// Two commands are supported:
//   1. `train` — trains the model on .docx documents
//   2. `ask`   — loads a checkpoint and answers a question
//
// Reference: Rust Book §7 (Modules), §12 (CLI programs)

// Declare the commands submodule
pub mod commands;

use anyhow::Result;
use clap::Parser;
use commands::{Commands, TrainArgs, AskArgs};

/// The main CLI struct — clap reads the fields and generates
/// argument parsing code automatically via the Parser derive macro.
#[derive(Parser, Debug)]
#[command(
    name = "word-doc-qa",
    version = "0.1.0",
    about = "Train a transformer Q&A model on .docx files, then ask questions."
)]
pub struct Cli {
    /// The subcommand to run (train or ask)
    #[command(subcommand)]
    pub command: Commands,
}

impl Cli {
    /// Match on the subcommand and dispatch to the correct use case.
    /// This keeps the CLI layer thin — it only routes, never computes.
    pub fn run(self) -> Result<()> {
        match self.command {
            Commands::Train(args) => self.run_train(args),
            Commands::Ask(args)   => self.run_ask(args),
        }
    }

    /// Handles the `train` subcommand.
    /// Converts CLI args into a TrainConfig and hands off to Layer 2.
    fn run_train(&self, args: TrainArgs) -> Result<()> {
        use crate::application::train_use_case::TrainUseCase;

        tracing::info!("Starting training on documents in: {}", args.docs_dir);

        // Convert CLI args → application config (separates presentation from domain)
        let use_case = TrainUseCase::new(args.into());
        use_case.execute()?;

        println!("Training complete. Checkpoint saved.");
        Ok(())
    }

    /// Handles the `ask` subcommand.
    /// Loads the model from checkpoint and prints the predicted answer.
    fn run_ask(&self, args: AskArgs) -> Result<()> {
        use crate::application::ask_use_case::AskUseCase;

        // Build the use case with checkpoint and docs directory paths
        let use_case = AskUseCase::new(
            args.checkpoint_dir.clone(),
            args.docs_dir.clone()
        )?;

        // Run inference and print the result
        let answer = use_case.answer(&args.question)?;
        println!("\nAnswer: {}", answer);
        Ok(())
    }
}
