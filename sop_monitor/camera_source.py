"""摄像头取流后端抽象。

本文件把客户端和具体取流方式解耦：界面只依赖统一的 FrameSource 接口，
后续接海康 HCNetSDK 时只需要新增/完善对应后端，不需要把 SDK 逻辑写进界面。
当前已实现 OpenCV 后端；RTSP 源会使用后台线程持续读流，只保留最新帧，避免
YOLO 推理较慢时旧帧排队造成越来越高的显示延迟。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sop_monitor.camera_utils import is_rtsp_source, open_camera


class FrameSource(Protocol):
    """统一视频源接口。"""

    def read(self) -> tuple[bool, object | None]:
        """读取一帧 BGR 图像。"""

    def release(self) -> None:
        """释放底层视频资源。"""


@dataclass(frozen=True)
class CameraSourceSpec:
    """创建视频源所需的运行参数。"""

    backend: str
    source: str
    width: int | None = None
    height: int | None = None
    hikvision_ip: str | None = None
    hikvision_user: str = "admin"
    hikvision_password: str | None = None
    hikvision_port: int = 8000
    hikvision_channel: str = "101"
    hikvision_sdk_dir: str = "third_party/hikvision"


def create_frame_source(spec: CameraSourceSpec) -> FrameSource:
    """根据配置创建具体取流后端。"""

    if spec.backend == "opencv":
        if is_rtsp_source(spec.source):
            return LatestFrameOpenCvSource(spec.source, spec.width, spec.height)
        return OpenCvFrameSource(spec.source, spec.width, spec.height)
    if spec.backend == "hikvision-sdk":
        return HikvisionSdkFrameSource(spec)
    raise ValueError(f"不支持的摄像头后端：{spec.backend}")


class OpenCvFrameSource:
    """普通 OpenCV 视频源，适合本地摄像头和离线视频文件。"""

    def __init__(self, source: str, width: int | None, height: int | None):
        self.capture = open_camera(source, width, height)

    def read(self) -> tuple[bool, object | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class LatestFrameOpenCvSource:
    """OpenCV RTSP 低延迟读取器。

    RTSP 监控只关心最新画面。后台线程持续读取并覆盖缓存，前台推理线程来取时
    直接拿最新帧；如果推理来不及，会自然丢掉旧帧，避免延迟堆积。
    """

    def __init__(self, source: str, width: int | None, height: int | None):
        self.capture = open_camera(source, width, height)
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, name="rtsp-latest-frame-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            with self._lock:
                self._latest_ok = ok
                self._latest_frame = frame if ok else None
            if not ok:
                time.sleep(0.03)

    def read(self) -> tuple[bool, object | None]:
        with self._lock:
            if not self._latest_ok or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self) -> None:
        self._running = False
        self._reader.join(timeout=1.0)
        self.capture.release()


class HikvisionSdkFrameSource:
    """海康 HCNetSDK 取流后端。

    本实现只在 Windows + 海康 SDK DLL 环境下可用。它使用 HCNetSDK 登录相机，
    RealPlay_V40 接收压缩码流，再用 PlayCtrl 解码成 BGR 帧。对外仍然只提供
    read/release，客户端不需要知道 SDK 细节。
    """

    def __init__(self, spec: CameraSourceSpec):
        import ctypes
        import os
        import platform
        from ctypes import wintypes

        if platform.system() != "Windows":
            raise RuntimeError("hikvision-sdk 后端只能在 Windows 上运行。当前系统请使用 --camera-backend opencv。")
        if not spec.hikvision_ip or not spec.hikvision_password:
            raise RuntimeError("使用 hikvision-sdk 后端时必须提供 --hikvision-ip 和 --hikvision-password。")

        self.spec = spec
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._released = False
        self._real_handle = -1
        self._user_id = -1
        self._play_port = ctypes.c_long(-1)

        sdk_dir = Path(spec.hikvision_sdk_dir).resolve()
        if not sdk_dir.exists():
            raise RuntimeError(f"找不到海康 SDK DLL 目录：{sdk_dir}")
        os.add_dll_directory(str(sdk_dir))

        self._hcnetsdk = ctypes.WinDLL(str(sdk_dir / "HCNetSDK.dll"))
        self._playctrl = ctypes.WinDLL(str(sdk_dir / "PlayCtrl.dll"))
        self._setup_types()
        self._setup_functions()
        self._init_sdk()
        self._login()
        self._start_realplay()

    def _setup_types(self) -> None:
        ctypes = self._ctypes
        wintypes = self._wintypes

        class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
            _fields_ = [
                ("sSerialNumber", ctypes.c_byte * 48),
                ("byAlarmInPortNum", ctypes.c_byte),
                ("byAlarmOutPortNum", ctypes.c_byte),
                ("byDiskNum", ctypes.c_byte),
                ("byDVRType", ctypes.c_byte),
                ("byChanNum", ctypes.c_byte),
                ("byStartChan", ctypes.c_byte),
                ("byAudioChanNum", ctypes.c_byte),
                ("byIPChanNum", ctypes.c_byte),
                ("byZeroChanNum", ctypes.c_byte),
                ("byMainProto", ctypes.c_byte),
                ("bySubProto", ctypes.c_byte),
                ("bySupport", ctypes.c_byte),
                ("bySupport1", ctypes.c_byte),
                ("bySupport2", ctypes.c_byte),
                ("wDevType", ctypes.c_ushort),
                ("bySupport3", ctypes.c_byte),
                ("byMultiStreamProto", ctypes.c_byte),
                ("byStartDChan", ctypes.c_byte),
                ("byStartDTalkChan", ctypes.c_byte),
                ("byHighDChanNum", ctypes.c_byte),
                ("bySupport4", ctypes.c_byte),
                ("byLanguageType", ctypes.c_byte),
                ("byVoiceInChanNum", ctypes.c_byte),
                ("byStartVoiceInChanNo", ctypes.c_byte),
                ("bySupport5", ctypes.c_byte),
                ("bySupport6", ctypes.c_byte),
                ("byMirrorChanNum", ctypes.c_byte),
                ("wStartMirrorChanNo", ctypes.c_ushort),
                ("bySupport7", ctypes.c_byte),
                ("byRes2", ctypes.c_byte),
            ]

        class NET_DVR_DEVICEINFO_V40(ctypes.Structure):
            _fields_ = [
                ("struDeviceV30", NET_DVR_DEVICEINFO_V30),
                ("bySupportLock", ctypes.c_byte),
                ("byRetryLoginTime", ctypes.c_byte),
                ("byPasswordLevel", ctypes.c_byte),
                ("byProxyType", ctypes.c_byte),
                ("dwSurplusLockTime", ctypes.c_uint),
                ("byCharEncodeType", ctypes.c_byte),
                ("bySupportDev5", ctypes.c_byte),
                ("byLoginMode", ctypes.c_byte),
                ("byRes2", ctypes.c_byte * 253),
            ]

        class NET_DVR_USER_LOGIN_INFO(ctypes.Structure):
            _fields_ = [
                ("sDeviceAddress", ctypes.c_char * 129),
                ("byUseTransport", ctypes.c_byte),
                ("wPort", ctypes.c_ushort),
                ("sUserName", ctypes.c_char * 64),
                ("sPassword", ctypes.c_char * 64),
                ("cbLoginResult", ctypes.c_void_p),
                ("pUser", ctypes.c_void_p),
                ("bUseAsynLogin", ctypes.c_bool),
                ("byProxyType", ctypes.c_byte),
                ("byUseUTCTime", ctypes.c_byte),
                ("byLoginMode", ctypes.c_byte),
                ("byHttps", ctypes.c_byte),
                ("iProxyID", ctypes.c_int),
                ("byVerifyMode", ctypes.c_byte),
                ("byRes3", ctypes.c_byte * 119),
            ]

        class NET_DVR_PREVIEWINFO(ctypes.Structure):
            _fields_ = [
                ("lChannel", ctypes.c_long),
                ("dwStreamType", ctypes.c_uint),
                ("dwLinkMode", ctypes.c_uint),
                ("hPlayWnd", wintypes.HWND),
                ("bBlocked", ctypes.c_bool),
                ("bPassbackRecord", ctypes.c_bool),
                ("byPreviewMode", ctypes.c_byte),
                ("byStreamID", ctypes.c_byte * 32),
                ("byProtoType", ctypes.c_byte),
                ("byRes1", ctypes.c_byte),
                ("byVideoCodingType", ctypes.c_byte),
                ("dwDisplayBufNum", ctypes.c_uint),
                ("byNPQMode", ctypes.c_byte),
                ("byRecvMetaData", ctypes.c_byte),
                ("byDataType", ctypes.c_byte),
                ("byRes", ctypes.c_byte * 213),
            ]

        class FRAME_INFO(ctypes.Structure):
            _fields_ = [
                ("nWidth", ctypes.c_long),
                ("nHeight", ctypes.c_long),
                ("nStamp", ctypes.c_long),
                ("nType", ctypes.c_long),
                ("nFrameRate", ctypes.c_long),
                ("dwFrameNum", ctypes.c_uint),
            ]

        self.NET_DVR_USER_LOGIN_INFO = NET_DVR_USER_LOGIN_INFO
        self.NET_DVR_DEVICEINFO_V40 = NET_DVR_DEVICEINFO_V40
        self.NET_DVR_PREVIEWINFO = NET_DVR_PREVIEWINFO
        self.FRAME_INFO = FRAME_INFO
        self.REALDATACALLBACK = ctypes.WINFUNCTYPE(
            None,
            ctypes.c_long,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_uint,
            ctypes.c_void_p,
        )
        self.DECCBFUNWIN = ctypes.WINFUNCTYPE(
            None,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_long,
            ctypes.POINTER(FRAME_INFO),
            ctypes.c_void_p,
            ctypes.c_long,
        )

    def _setup_functions(self) -> None:
        ctypes = self._ctypes

        self._hcnetsdk.NET_DVR_Init.restype = ctypes.c_bool
        self._hcnetsdk.NET_DVR_Cleanup.restype = ctypes.c_bool
        self._hcnetsdk.NET_DVR_GetLastError.restype = ctypes.c_uint
        self._hcnetsdk.NET_DVR_SetConnectTime.argtypes = [ctypes.c_uint, ctypes.c_uint]
        self._hcnetsdk.NET_DVR_SetReconnect.argtypes = [ctypes.c_uint, ctypes.c_bool]
        self._hcnetsdk.NET_DVR_Login_V40.argtypes = [
            ctypes.POINTER(self.NET_DVR_USER_LOGIN_INFO),
            ctypes.POINTER(self.NET_DVR_DEVICEINFO_V40),
        ]
        self._hcnetsdk.NET_DVR_Login_V40.restype = ctypes.c_long
        self._hcnetsdk.NET_DVR_Logout.argtypes = [ctypes.c_long]
        self._hcnetsdk.NET_DVR_RealPlay_V40.argtypes = [
            ctypes.c_long,
            ctypes.POINTER(self.NET_DVR_PREVIEWINFO),
            self.REALDATACALLBACK,
            ctypes.c_void_p,
        ]
        self._hcnetsdk.NET_DVR_RealPlay_V40.restype = ctypes.c_long
        self._hcnetsdk.NET_DVR_StopRealPlay.argtypes = [ctypes.c_long]

        self._playctrl.PlayM4_GetPort.argtypes = [ctypes.POINTER(ctypes.c_long)]
        self._playctrl.PlayM4_GetPort.restype = ctypes.c_bool
        self._playctrl.PlayM4_FreePort.argtypes = [ctypes.c_long]
        self._playctrl.PlayM4_SetStreamOpenMode.argtypes = [ctypes.c_long, ctypes.c_uint]
        self._playctrl.PlayM4_OpenStream.argtypes = [
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self._playctrl.PlayM4_SetDecCallBack.argtypes = [ctypes.c_long, self.DECCBFUNWIN]
        self._playctrl.PlayM4_Play.argtypes = [ctypes.c_long, ctypes.c_void_p]
        self._playctrl.PlayM4_InputData.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint]
        self._playctrl.PlayM4_Stop.argtypes = [ctypes.c_long]
        self._playctrl.PlayM4_CloseStream.argtypes = [ctypes.c_long]

    def _init_sdk(self) -> None:
        if not self._hcnetsdk.NET_DVR_Init():
            raise RuntimeError(f"NET_DVR_Init 失败，错误码：{self._last_sdk_error()}")
        self._hcnetsdk.NET_DVR_SetConnectTime(2000, 1)
        self._hcnetsdk.NET_DVR_SetReconnect(1000, True)

    def _login(self) -> None:
        ctypes = self._ctypes

        login_info = self.NET_DVR_USER_LOGIN_INFO()
        login_info.sDeviceAddress = self.spec.hikvision_ip.encode("utf-8")
        login_info.wPort = self.spec.hikvision_port
        login_info.sUserName = self.spec.hikvision_user.encode("utf-8")
        login_info.sPassword = (self.spec.hikvision_password or "").encode("utf-8")
        login_info.bUseAsynLogin = False

        device_info = self.NET_DVR_DEVICEINFO_V40()
        self._user_id = self._hcnetsdk.NET_DVR_Login_V40(ctypes.byref(login_info), ctypes.byref(device_info))
        if self._user_id < 0:
            raise RuntimeError(f"海康 SDK 登录失败，错误码：{self._last_sdk_error()}")

    def _start_realplay(self) -> None:
        ctypes = self._ctypes

        if not self._playctrl.PlayM4_GetPort(ctypes.byref(self._play_port)):
            raise RuntimeError("PlayM4_GetPort 失败，无法分配播放库端口。")

        preview_info = self.NET_DVR_PREVIEWINFO()
        preview_info.lChannel, preview_info.dwStreamType = self._parse_channel(self.spec.hikvision_channel)
        preview_info.dwLinkMode = 0
        preview_info.hPlayWnd = None
        preview_info.bBlocked = False
        preview_info.dwDisplayBufNum = 1

        self._decode_callback = self.DECCBFUNWIN(self._on_decoded_frame)
        self._realdata_callback = self.REALDATACALLBACK(self._on_real_data)
        self._real_handle = self._hcnetsdk.NET_DVR_RealPlay_V40(
            self._user_id,
            ctypes.byref(preview_info),
            self._realdata_callback,
            None,
        )
        if self._real_handle < 0:
            raise RuntimeError(f"NET_DVR_RealPlay_V40 失败，错误码：{self._last_sdk_error()}")

    def _on_real_data(self, real_handle, data_type, buffer, buffer_size, user) -> None:
        # 1 = NET_DVR_SYSHEAD，2 = NET_DVR_STREAMDATA。
        if self._released or not buffer or buffer_size <= 0:
            return
        if data_type == 1:
            self._playctrl.PlayM4_SetStreamOpenMode(self._play_port.value, 0)
            opened = self._playctrl.PlayM4_OpenStream(self._play_port.value, buffer, buffer_size, 2 * 1024 * 1024)
            if opened:
                self._playctrl.PlayM4_SetDecCallBack(self._play_port.value, self._decode_callback)
                self._playctrl.PlayM4_Play(self._play_port.value, None)
        elif data_type == 2:
            self._playctrl.PlayM4_InputData(self._play_port.value, buffer, buffer_size)

    def _on_decoded_frame(self, port, frame_buffer, frame_size, frame_info, reserved1, reserved2) -> None:
        if self._released or not frame_buffer or not frame_info:
            return

        import cv2
        import numpy as np

        width = int(frame_info.contents.nWidth)
        height = int(frame_info.contents.nHeight)
        frame_type = int(frame_info.contents.nType)
        if width <= 0 or height <= 0 or frame_size <= 0:
            return

        raw = self._ctypes.string_at(frame_buffer, frame_size)
        try:
            if frame_type == 3 and frame_size >= width * height * 3 // 2:
                yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YV12)
            elif frame_size >= width * height * 3:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            else:
                return
        except ValueError:
            return

        with self._lock:
            self._latest_frame = frame
            self._latest_ok = True

    def read(self) -> tuple[bool, object | None]:
        with self._lock:
            if not self._latest_ok or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._real_handle >= 0:
            self._hcnetsdk.NET_DVR_StopRealPlay(self._real_handle)
            self._real_handle = -1
        if self._play_port.value >= 0:
            self._playctrl.PlayM4_Stop(self._play_port.value)
            self._playctrl.PlayM4_CloseStream(self._play_port.value)
            self._playctrl.PlayM4_FreePort(self._play_port.value)
            self._play_port.value = -1
        if self._user_id >= 0:
            self._hcnetsdk.NET_DVR_Logout(self._user_id)
            self._user_id = -1
        self._hcnetsdk.NET_DVR_Cleanup()

    def _last_sdk_error(self) -> int:
        return int(self._hcnetsdk.NET_DVR_GetLastError())

    @staticmethod
    def _parse_channel(channel: str) -> tuple[int, int]:
        """把 RTSP 风格通道号转成 SDK 通道号和码流类型。

        海康常见 RTSP 通道 101 表示 1 通道主码流，102 表示 1 通道子码流。
        SDK 中 lChannel=1，dwStreamType=0/1。
        """

        text = str(channel).strip()
        if text.isdigit() and len(text) >= 3:
            sdk_channel = int(text[:-2])
            stream_suffix = text[-2:]
            stream_type = 1 if stream_suffix == "02" else 0
            return sdk_channel, stream_type
        return int(text), 0
