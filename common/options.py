import argparse

def make_options(*args, opts_override={}, **kwargs):
    parser = argparse.ArgumentParser()
    for arg in args: arg(parser)
    group = parser.add_argument_group('default', conflict_handler='resolve')
    for k,v in kwargs.items():
        if isinstance(v, bool): group.add_argument(f'--{k}', action='store_true', default=v)
        elif isinstance(v, (list, tuple)): group.add_argument(f'--{k}', nargs='*', default=v)
        else: group.add_argument(f'--{k}', type=v.__class__, default=v)
    opts, _ = parser.parse_known_args()
    for k, v in opts_override.items():
        setattr(opts, k, v)
    return opts
