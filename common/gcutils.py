import os, asyncio, threading, ray, inspect, functools, glob, shutil

def event_loop_init():
    try:
        import nest_asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        nest_asyncio.apply(asyncio.get_event_loop())
    except:
        pass

def run_async(target, args=tuple()):
    thread = threading.Thread(target=target, args=args)
    thread.start()
    return thread

def ensure_async(method):
    if isinstance(method, ray.remote_function.RemoteFunction):
        return method.remote
    elif inspect.iscoroutinefunction(method):
        return method
    else:
        return method

def when_any(coroutines, interval=None):
    coroutines = [asyncio.ensure_future(x, loop=asyncio.get_event_loop()) for x in coroutines]
    task = asyncio.wait(coroutines, timeout=interval, return_when=asyncio.FIRST_COMPLETED)
    ready, not_ready = when(task)
    results = [x.result() for x in ready]
    return results, list(not_ready)

def when_all(coroutines):
    return when(asyncio.gather(*coroutines))

def when(task):
    return asyncio.get_event_loop().run_until_complete(task)

def when_ordered(coroutines, interval=None):
    not_ready = [asyncio.ensure_future(x, loop=asyncio.get_event_loop()) for x in coroutines]
    tasks = {k: x for k,x in enumerate(not_ready)}
    done = {k: False for k in range(len(not_ready))}
    idx = {x: k for k,x in enumerate(not_ready)}
    current = 0
    while not_ready:
        task = asyncio.wait(not_ready, timeout=interval, return_when=asyncio.FIRST_COMPLETED)
        ready, not_ready = when(task)
        for task in ready:
            k = idx[task]
            done[k] = True
        while current in done and done[current]:
            yield tasks[current].result()
            current += 1

def as_async(method):
    def _method(*args, **kwargs):
        bind = functools.partial(method, *args, **kwargs)
        return asyncio.get_event_loop().run_in_executor(None, bind)
    return _method

async def wait(coroutines, timeout=None):
    if not isinstance(coroutines, (list, tuple)):
        return await asyncio.wait_for(coroutines, timeout=timeout)
    coroutines = [asyncio.ensure_future(x, loop=asyncio.get_event_loop()) for x in coroutines]
    if len(coroutines) == 0: return [], []
    ready, not_ready = await asyncio.wait(coroutines, timeout=timeout)
    return [x.result() for x in ready], not_ready

def list_files(folder):
    if os.path.exists(folder):
        return [x for x in os.listdir(folder) if os.path.isfile(os.path.join(folder, x))]
    else:
        return []

def list_filepaths(folder, pattern='*.*'):
    if isinstance(pattern, str): pattern = pattern,
    paths = []
    for x in pattern: paths.extend(glob.glob(f'{glob.escape(folder)}/{x}'))
    return paths

def recursive_list_files(folder, pattern='*'):
    from pathlib import Path
    if isinstance(pattern, str): pattern = pattern,
    paths = []
    for x in pattern: paths.extend(Path(folder).glob(f'**/{x}'))
    return [str(x) for x in paths if x.is_file()]

def recursive_list_folder(folder, pattern='*'):
    from pathlib import Path
    paths = Path(folder).glob(f'**/{pattern}')
    return [str(x) for x in paths if x.is_dir()]

def list_folders(folder):
    if not os.path.isdir(folder): return []
    return [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]

def list_folder_paths(folder):
    return [os.path.join(folder, x) for x in list_folders(folder)]

def basename(path):
    return os.path.basename(path)

def extension(path):
    return os.path.splitext(path)[-1][1:].lower()

def basename_without_extension(path):
    return os.path.splitext(basename(path))[0]

def replace_extension(path, new_extension):
    old_extension = extension(path)
    return path[:-len(old_extension)] + new_extension

def add_suffix(path, suffix):
    name = basename_without_extension(path)
    name = name + suffix + '.' + extension(path)
    return os.path.join(os.path.dirname(path), name)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def move(source, target):
    ensure_folder(target)
    shutil.move(source, target)

def copy_file(source_path, target_path):
    ensure_folder(target_path)
    shutil.copyfile(source_path, target_path)

def ensure_folder(filename):
    folder = os.path.dirname(os.path.abspath(filename))
    mkdir(folder)
    return folder

def join(*sections):
    return os.path.join(*sections)

def write_all_lines(filename, lines, encoding="utf8"):
    ensure_folder(filename)
    with open(filename, "w", encoding=encoding) as file:
        for line in lines:
            file.write(line + "\n")

def read_all_lines(filename, encoding="utf8", skip_empty_lines=True):
    if not os.path.isfile(filename):
        return
    with open(filename, encoding=encoding) as file:
        for line in file:
            line = line.rstrip().strip("\ufeff")
            if not skip_empty_lines or line:
                yield line

def write_all_lines(filename, lines, encoding="utf8"):
    ensure_folder(filename)
    with open(filename, "w", encoding=encoding) as file:
        for line in lines:
            file.write(line + "\n")

def read_all_text(path, encoding='utf8'):
    with open(path, "r", encoding=encoding, errors='ignore') as file:
        return file.read().strip("\ufeff").replace("\r\n", "\n")

def write_all_text(path, text, encoding="utf8"):
    ensure_folder(path)
    with open(path, "w", encoding=encoding) as file:
        file.write(text)

def read_all_bytes(path):
    with open(path, "rb") as file:
        return file.read()

def write_all_bytes(path, bytes):
    ensure_folder(path)
    with open(path, "wb") as file:
        file.write(bytes)

def write_file(path, content):
    if isinstance(content, bytes):
        write_all_bytes(path, content)
    else:
        write_all_text(path, content)

def read_file(path: str):
    if is_text_file(path):
        return read_all_text(path)
    else:
        return read_all_bytes(path)

def is_text_file(path):
    return extension(path) in ["txt", "html", "xml", "json"]
