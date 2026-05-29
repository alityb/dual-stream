__all__ = ["DualStreamAgent"]


def __getattr__(name: str):
    if name == "DualStreamAgent":
        from dual_stream.agent import DualStreamAgent

        return DualStreamAgent
    raise AttributeError(name)
