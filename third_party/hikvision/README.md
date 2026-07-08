# 海康 Windows SDK 文件目录

把海康 Windows 64 位 SDK 的运行 DLL 放在本目录，客户端使用
`--camera-backend hikvision-sdk` 时会从这里加载。

常见文件包括：

```text
HCNetSDK.dll
PlayCtrl.dll
HCCore.dll
hpr.dll
libcrypto-*.dll
libssl-*.dll
其他 SDK 随包 DLL
```

启动示例：

```bat
python -m sop_monitor.desktop_app ^
  --camera-backend hikvision-sdk ^
  --hikvision-sdk-dir third_party\hikvision ^
  --hikvision-ip 192.168.114.222 ^
  --hikvision-user admin ^
  --hikvision-password 密码 ^
  --hikvision-port 8000 ^
  --hikvision-channel 102 ^
  --config configs\calibrated_sop.json ^
  --model runs\installed_part_roi\weights\best.pt ^
  --conf 0.35
```

`--hikvision-channel 101` 表示 1 通道主码流，`102` 表示 1 通道子码流。
