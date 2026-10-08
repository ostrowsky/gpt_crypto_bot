"""Read a stable Windows file handle while permitting atomic dataset replacement."""
from contextlib import contextmanager
import ctypes,io,os,time


def publish_snapshot(source,destination):
    """ReplaceFile preserves ACLs and permits shared-delete readers on Windows."""
    if os.name!='nt' or not destination.exists():return source.replace(destination)
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);replace=kernel.ReplaceFileW
    replace.argtypes=(wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.DWORD,wintypes.LPVOID,wintypes.LPVOID);replace.restype=wintypes.BOOL
    if not replace(str(destination.resolve()),str(source.resolve()),None,0,None,None):raise ctypes.WinError(ctypes.get_last_error())


@contextmanager
def open_snapshot(path,mode='r',encoding='utf-8',errors='strict',retry_seconds=30):
    if mode not in ('r','rb'):raise ValueError('read-only snapshots only')
    if os.name!='nt':
        with path.open(mode,**({} if mode=='rb' else dict(encoding=encoding,errors=errors))) as stream:yield stream
        return
    import msvcrt
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW;create.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE);create.restype=wintypes.HANDLE
    close=kernel.CloseHandle;close.argtypes=(wintypes.HANDLE,);close.restype=wintypes.BOOL
    deadline=time.monotonic()+retry_seconds
    while True:
        # READ|WRITE|DELETE sharing keeps the reader on its original inode while
        # cooperative writers atomically publish a replacement path.
        handle=create(str(path.resolve()),0x80000000,0x7,None,3,0x80,None)
        if handle!=ctypes.c_void_p(-1).value:break
        code=ctypes.get_last_error()
        if code not in (32,33) or time.monotonic()>=deadline:raise ctypes.WinError(code)
        time.sleep(.05)
    try:fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except BaseException:close(handle);raise
    with os.fdopen(fd,'rb') as raw:
        if mode=='rb':yield raw
        else:
            with io.TextIOWrapper(raw,encoding=encoding,errors=errors) as text:yield text
