import functools


def serialize_fdl(config):
    import fiddle as fdl

    if isinstance(config, fdl.Buildable):
        result = {
            "__type__": type(config).__name__,
            "__fn_or_cls__": str(config.__fn_or_cls__),
        }
        for k, v in config.__arguments__.items():
            try:
                result[k] = serialize_fdl(v)
            except Exception:
                result[k] = f"<non-serializable: {type(v).__name__}>"
        return result
    elif isinstance(config, (list, tuple)):
        return [serialize_fdl(x) for x in config]
    elif isinstance(config, dict):
        return {k: serialize_fdl(v) for k, v in config.items()}
    elif isinstance(config, (str, int, float, bool, type(None))):
        return config
    else:
        # Fallback for non-serializable objects
        return f"<non-serializable: {type(config).__name__}>"


def deep_debug(obj, name="obj", indent=0):
    if indent > 2:
        return
    pad = " " * indent
    print(f"{pad}{name}: {type(obj)}")

    if isinstance(obj, functools.partial):
        print(f"{pad}  partial.func = {obj.func}")
        for k, v in obj.keywords.items():
            deep_debug(v, k, indent + 1)
        return

    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                print(f"{pad}  {k} = {v}")
            else:
                deep_debug(v, k, indent + 1)
