"""Shallow MLP modeling attack against a single APUF."""

from attacks.neural_net import run_attack


def main() -> None:
    # MLP baseline: one hidden layer with 64 neurons.
    run_attack("MLP", hidden_layers=(64,))


if __name__ == "__main__":
    main()
