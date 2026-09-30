"""Deeper neural-network modeling attack against a single APUF."""

from attacks.neural_net import run_attack


def main() -> None:
    # DNN baseline: three hidden layers, making it deeper than the MLP baseline.
    run_attack("DNN", hidden_layers=(128, 64, 32))


if __name__ == "__main__":
    main()
