"""A second real torch engine — MLP regression on sklearn's diabetes dataset.

A different domain (regression, RMSE metric — not classification error) sharing the SAME
orchestrator: proof that adding a domain is just a profile + an (editable train, frozen prepare)
engine pair, with zero changes to the loop, the agents, or the guards.
"""
