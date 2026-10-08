"""Train the MosaiChunk descriptor encoder by self-distillation."""

from mosaichunk.cli import main

if __name__ == "__main__":
    main(training=True)
