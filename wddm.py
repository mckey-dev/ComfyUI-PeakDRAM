"""タスクマネージャーと同じ系統のメモリを Windows から読む。

DRAM はシステム全体の使用中物理メモリ。
専用DRAM / 共有DRAM は、専用容量が最大の GPU アダプタの
Dedicated Usage / Shared Usage（GPU Adapter Memory）。
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from dataclasses import dataclass

_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2
_PDH_CSTATUS_VALID_DATA = 0
_PDH_CSTATUS_NEW_DATA = 1

_lock = threading.Lock()
_adapter: AdapterInfo | None = None
_pdh_query = ctypes.c_void_p()
_pdh_dedicated = ctypes.c_void_p()
_pdh_shared = ctypes.c_void_p()
_pdh_ready = False


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    luid_high: int
    luid_low: int
    dedicated_bytes: int
    shared_bytes: int

    @property
    def luid_token(self) -> str:
        return f"luid_0x{self.luid_high & 0xFFFFFFFF:08X}_0x{self.luid_low & 0xFFFFFFFF:08X}".lower()


@dataclass(frozen=True)
class MemorySample:
    dram_bytes: int
    dram_total_bytes: int
    dedicated_bytes: int
    shared_bytes: int
    adapter: AdapterInfo


@dataclass
class PeakReport:
    samples: int
    dram_bytes: int = 0
    dram_start_bytes: int = 0
    dram_total_bytes: int = 0
    dedicated_bytes: int = 0
    dedicated_start_bytes: int = 0
    shared_bytes: int = 0
    shared_start_bytes: int = 0
    adapter_name: str = ""
    dedicated_total_bytes: int = 0
    shared_total_bytes: int = 0
    error: str = ""

    def observe(self, sample: MemorySample) -> None:
        if self.samples == 0:
            self.dram_start_bytes = sample.dram_bytes
            self.dedicated_start_bytes = sample.dedicated_bytes
            self.shared_start_bytes = sample.shared_bytes
        self.samples += 1
        self.dram_bytes = max(self.dram_bytes, sample.dram_bytes)
        self.dedicated_bytes = max(self.dedicated_bytes, sample.dedicated_bytes)
        self.shared_bytes = max(self.shared_bytes, sample.shared_bytes)
        self.dram_total_bytes = sample.dram_total_bytes
        self.adapter_name = sample.adapter.name
        self.dedicated_total_bytes = sample.adapter.dedicated_bytes
        self.shared_total_bytes = sample.adapter.shared_bytes

    def line(self, prompt_id: str, status: str, elapsed_sec: float) -> str:
        if self.samples == 0:
            reason = self.error or "サンプルがありません"
            return f"[PeakDRAM] prompt={prompt_id} {status} {elapsed_sec:.2f}s | 計測できませんでした: {reason}"
        return (
            f"[PeakDRAM] prompt={prompt_id} {status} {elapsed_sec:.2f}s | "
            f"{_format_usage_with_delta(self.dram_bytes, self.dram_total_bytes, self.dram_start_bytes, 'DRAM')} | "
            f"{_format_usage_with_delta(self.dedicated_bytes, self.dedicated_total_bytes, self.dedicated_start_bytes, '専用DRAM')} | "
            f"{_format_usage_with_delta(self.shared_bytes, self.shared_total_bytes, self.shared_start_bytes, '共有DRAM')} | "
            f"{self.adapter_name}"
        )


def startup_line() -> str:
    """拡張読み込み直後（ComfyUI 起動時）の現在値を 1 行にまとめる。"""
    try:
        sample = sample_once()
    except Exception as exc:
        return f"[PeakDRAM] 起動時 | 計測できませんでした: {exc}"
    if sample is None:
        return "[PeakDRAM] 起動時 | 計測できませんでした: GPU アダプタまたは性能カウンタを読めません"
    return (
        f"[PeakDRAM] 起動時 | "
        f"{_format_usage(sample.dram_bytes, sample.dram_total_bytes, 'DRAM')} | "
        f"{_format_usage(sample.dedicated_bytes, sample.adapter.dedicated_bytes, '専用DRAM')} | "
        f"{_format_usage(sample.shared_bytes, sample.adapter.shared_bytes, '共有DRAM')} | "
        f"{sample.adapter.name}"
    )


def _format_usage(used_bytes: int, total_bytes: int, label: str) -> str:
    return f"{label} {_mib(used_bytes)}/{_mib(total_bytes)} MiB"


def _format_usage_with_delta(peak_bytes: int, total_bytes: int, start_bytes: int, label: str) -> str:
    return f"{label} {_mib(peak_bytes)}/{_mib(total_bytes)} MiB (+{_mib(peak_bytes - start_bytes)})"


class PeakSampler:
    """実行中にメモリを刻み、各項目の最大値を残す。"""

    def __init__(self, interval_sec: float = 0.1):
        self.interval_sec = interval_sec
        self.report = PeakReport(samples=0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._observe()
        self._thread = threading.Thread(target=self._run, name="peak-dram", daemon=True)
        self._thread.start()

    def stop(self) -> PeakReport:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._observe()
        return self.report

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._observe()

    def _observe(self) -> None:
        try:
            sample = sample_once()
        except Exception as exc:
            if not self.report.error:
                self.report.error = str(exc)
            return
        if sample is None:
            if not self.report.error:
                self.report.error = "GPU アダプタまたは性能カウンタを読めません"
            return
        self.report.observe(sample)


def sample_once() -> MemorySample | None:
    """現在の DRAM / 専用DRAM / 共有DRAM を返す。Windows 以外は None。"""
    if sys.platform != "win32":
        return None
    with _lock:
        adapter = _selected_adapter()
        if adapter is None or not _ensure_pdh():
            return None
        dedicated, shared = _read_adapter_usage(adapter)
    dram_used, dram_total = _physical_memory()
    return MemorySample(
        dram_bytes=dram_used,
        dram_total_bytes=dram_total,
        dedicated_bytes=dedicated,
        shared_bytes=shared,
        adapter=adapter,
    )


def _mib(num_bytes: int) -> str:
    return str(int(round(num_bytes / (1024 * 1024))))


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG),
    ]


class _DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("doubleValue", ctypes.c_double),
    ]


class _PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
    _fields_ = [
        ("szName", ctypes.c_wchar_p),
        ("FmtValue", _PDH_FMT_COUNTERVALUE),
    ]


def _make_guid(text: str) -> _GUID:
    raw = bytes.fromhex(text.strip("{}").replace("-", ""))
    guid = _GUID()
    guid.Data1 = int.from_bytes(raw[0:4], "big")
    guid.Data2 = int.from_bytes(raw[4:6], "big")
    guid.Data3 = int.from_bytes(raw[6:8], "big")
    guid.Data4[:] = raw[8:16]
    return guid


_IID_FACTORY1 = _make_guid("770AAE78-F26F-4DBA-A829-253C83D1B387")


def _physical_memory() -> tuple[int, int]:
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    total = int(status.ullTotalPhys)
    used = total - int(status.ullAvailPhys)
    return used, total


def _selected_adapter() -> AdapterInfo | None:
    """専用容量が最大の実アダプタを一度だけ選ぶ。torch のデバイス名があればそれを優先する。"""
    global _adapter
    if _adapter is not None:
        return _adapter
    adapters = _enum_adapters()
    if not adapters:
        return None
    want = _torch_device_name()
    chosen = None
    if want:
        for adapter in adapters:
            if _names_match(want, adapter.name):
                chosen = adapter
                break
    if chosen is None:
        chosen = max(adapters, key=lambda item: item.dedicated_bytes)
    _adapter = chosen
    return chosen


def _torch_device_name() -> str:
    try:
        import torch
    except Exception:
        return ""
    try:
        if not torch.cuda.is_available():
            return ""
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return ""


def _enum_adapters() -> list[AdapterInfo]:
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx(None, 0)

    factory = ctypes.c_void_p()
    create = ctypes.windll.dxgi.CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory)) != 0 or not factory.value:
        return []

    found: list[AdapterInfo] = []
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        hr = _call(factory, 7, ctypes.c_uint, index, ctypes.c_void_p, ctypes.byref(adapter))
        if hr != 0:
            break
        desc = _DXGI_ADAPTER_DESC()
        if _call(adapter, 8, ctypes.c_void_p, ctypes.byref(desc)) == 0:
            name = str(desc.Description)
            dedicated = int(desc.DedicatedVideoMemory)
            if dedicated > 0 and "basic render" not in name.lower():
                found.append(
                    AdapterInfo(
                        name=name,
                        luid_high=int(desc.AdapterLuid.HighPart),
                        luid_low=int(desc.AdapterLuid.LowPart),
                        dedicated_bytes=dedicated,
                        shared_bytes=int(desc.SharedSystemMemory),
                    )
                )
        _call(adapter, 2)
        index += 1
    _call(factory, 2)
    return found


def _ensure_pdh() -> bool:
    global _pdh_ready
    if _pdh_ready:
        return True
    pdh = ctypes.windll.pdh
    pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    pdh.PdhOpenQueryW.restype = ctypes.c_uint32
    pdh.PdhAddEnglishCounterW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    pdh.PdhAddEnglishCounterW.restype = ctypes.c_uint32
    pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
    pdh.PdhCollectQueryData.restype = ctypes.c_uint32
    pdh.PdhGetFormattedCounterArrayW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_uint32
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(_pdh_query)) != 0:
        return False
    if pdh.PdhAddEnglishCounterW(
        _pdh_query,
        r"\GPU Adapter Memory(*)\Dedicated Usage",
        None,
        ctypes.byref(_pdh_dedicated),
    ) != 0:
        return False
    if pdh.PdhAddEnglishCounterW(
        _pdh_query,
        r"\GPU Adapter Memory(*)\Shared Usage",
        None,
        ctypes.byref(_pdh_shared),
    ) != 0:
        return False
    pdh.PdhCollectQueryData(_pdh_query)
    _pdh_ready = True
    return True


def _read_adapter_usage(adapter: AdapterInfo) -> tuple[int, int]:
    pdh = ctypes.windll.pdh
    pdh.PdhCollectQueryData(_pdh_query)
    dedicated = _sum_counter(_pdh_dedicated, adapter.luid_token)
    shared = _sum_counter(_pdh_shared, adapter.luid_token)
    return dedicated, shared


def _sum_counter(counter: ctypes.c_void_p, luid_token: str) -> int:
    pdh = ctypes.windll.pdh
    size = wintypes.DWORD(0)
    count = wintypes.DWORD(0)
    hr = pdh.PdhGetFormattedCounterArrayW(
        counter,
        _PDH_FMT_DOUBLE,
        ctypes.byref(size),
        ctypes.byref(count),
        None,
    )
    if (hr & 0xFFFFFFFF) not in (0, _PDH_MORE_DATA) or size.value == 0:
        return 0
    buf = (ctypes.c_byte * size.value)()
    hr = pdh.PdhGetFormattedCounterArrayW(
        counter,
        _PDH_FMT_DOUBLE,
        ctypes.byref(size),
        ctypes.byref(count),
        buf,
    )
    if hr != 0:
        return 0
    items = ctypes.cast(buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM))
    total = 0.0
    for index in range(count.value):
        item = items[index]
        name = (item.szName or "").lower()
        status = int(item.FmtValue.CStatus)
        if luid_token not in name:
            continue
        if status not in (_PDH_CSTATUS_VALID_DATA, _PDH_CSTATUS_NEW_DATA):
            continue
        total += float(item.FmtValue.doubleValue)
    return int(total)


def _names_match(want: str, have: str) -> bool:
    left = want.lower().strip()
    right = have.lower().strip()
    if not left:
        return False
    if left in right or right in left:
        return True
    for prefix in ("nvidia ", "amd ", "intel "):
        left = left.removeprefix(prefix)
        right = right.removeprefix(prefix)
    return bool(left) and (left in right or right in left)


def _call(obj: ctypes.c_void_p, slot: int, *typed_args) -> int:
    this = obj.value
    vtbl = ctypes.cast(this, ctypes.POINTER(ctypes.c_void_p))[0]
    fn = ctypes.cast(
        vtbl + slot * ctypes.sizeof(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )[0]
    types: list = [ctypes.c_void_p]
    values: list = [this]
    for index in range(0, len(typed_args), 2):
        types.append(typed_args[index])
        values.append(typed_args[index + 1])
    proto = ctypes.WINFUNCTYPE(ctypes.c_long, *types)
    return int(proto(fn)(*values))
