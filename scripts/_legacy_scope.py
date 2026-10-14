"""Keep historical study assumptions explicit at their public entrypoints."""


def require_static_study(config):
    """Protect static-sign diagnostics, fixed contacts, and chunk continuation."""
    if (config.ally_giver_fraction > 0 or config.opponent_giver_fraction > 0
            or config.conditional_ally_fraction > 0):
        raise ValueError(
            "Historical study scripts require uniform giving and static relationships; "
            "use the batch command or explorer for giver preferences and payment-dependent loyalty"
        )
