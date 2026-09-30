from pathlib import Path

from challenge_data import (
    generate_and_save_challenges,
    load_challenges,
)

# 生成并保存 Challenge。这里的 seed 固定后，每次结果都可复现。
npy_path = Path("data/challenges.npy")
challenges = generate_and_save_challenges(
    npy_path,
    num_challenges=100,
    challenge_length=64,
    seed=42,
)
# 后续可以直接读取之前保存的 NPY 文件。
challenges = load_challenges(npy_path)
print("Challenge 数组形状：", challenges.shape)


