"""面向过程版本的调用示例。"""

from challenge import generate_challenges


challenges = generate_challenges(
    num_challenges=100,
    challenge_length=64,
    seed=42,
)

print(challenges)
