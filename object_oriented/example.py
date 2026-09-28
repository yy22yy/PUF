"""面向对象版本的调用示例。"""

from challenge import ChallengeGenerator


generator = ChallengeGenerator(challenge_length=64, seed=42)
challenges = generator.generate(num_challenges=100)

print(challenges)
