"""Hydra config tree, shipped inside the package.

Not importable code — the YAML in here is data. The module marker exists
because Hydra resolves ``@hydra.main(config_path="conf")`` as a *module* when
the app lives in a package, and an editable install has no filesystem fallback:
without this file ``spifil-fit`` fails with "Primary config module 'spifil.conf'
not found".
"""
