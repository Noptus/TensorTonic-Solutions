def value_iteration_step(
    values: list,
    transitions: list,
    rewards: list,
    gamma: float
) -> list[float]:
    """Returns one updated floating-point value for every state."""
    updated_values = []

    for state in range(len(values)):
        action_values = []

        for action in range(len(rewards[state])):
            expected_next_value = sum(
                probability * values[next_state]
                for next_state, probability in enumerate(transitions[state][action])
            )

            q_value = rewards[state][action] + gamma * expected_next_value
            action_values.append(q_value)

        updated_values.append(float(max(action_values)))

    return updated_values