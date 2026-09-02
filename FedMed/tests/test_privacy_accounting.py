from flwr.serverapp import (
    PrivacyConfig,
    RdpAccountant,
    NeighboringRelation,
    SamplingMethod,
)
from flwr.supercore.privacy_accounting import GaussianPrivacyEvent


def test_fedmed_privacy_accounting():

    print("\n===== FEDMED PRIVACY ACCOUNTING =====")

    population_size = 3
    sample_size = 3
    num_rounds = 5
    target_delta = 1e-5

    noise_levels = [
        0.5,
        1.0,
        2.0,
    ]

    for noise_multiplier in noise_levels:

        print("\n----------------------------------------")
        print(f"Noise multiplier : {noise_multiplier}")

        config = PrivacyConfig(
            target_delta=target_delta,
            population_size=population_size,
            neighboring_relation=(
                NeighboringRelation.ADD_OR_REMOVE_ONE
            ),
            sampling_method=(
                SamplingMethod.NO_AMPLIFICATION
            ),
        )

        accountant = RdpAccountant(config)

        event = GaussianPrivacyEvent(
            noise_multiplier=noise_multiplier,
            sample_size=sample_size,
            population_size=population_size,
        )

        # Compose one private release for each federated round
        accountant.compose(
            event,
            count=num_rounds,
        )

        epsilon = accountant.get_epsilon(
            target_delta
        )

        privacy_spent = accountant.get_privacy_spent(
            target_delta
        )

        print(f"Rounds           : {num_rounds}")
        print(f"Population       : {population_size}")
        print(f"Sample size      : {sample_size}")
        print(f"Target delta     : {target_delta}")
        print(f"Epsilon          : {epsilon:.6f}")
        print(f"Privacy spent    : {privacy_spent}")

    print("\n===== ACCOUNTING TEST COMPLETE =====")

    assert epsilon > 0