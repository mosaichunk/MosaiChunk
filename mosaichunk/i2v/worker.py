"""Initialize the upstream import path before importing the I2V implementation."""

import sys

from mosaichunk.runtime import bootstrap, record_environment_on_exit

bootstrap("i2v")
record_environment_on_exit()
mode = sys.argv.pop(1)
if mode == "train":
    from .train import main
else:
    from .evalrun import main

if __name__ == "__main__":
    main()
