import win32com.client as windows

def get_usb_camera_devices():
    CAMERA_KEYWORDS = ["camera", "webcam", "usb video", "logi", "webcamera", "c270", "c310"]
    matched_devices = []

    wmi = windows.Dispatch("WbemScripting.SWbemLocator")
    service = wmi.ConnectServer(".", "root\\cimv2")
    devices = service.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE PNPDeviceID LIKE 'USB%'")

    for device in devices:
        name = device.Name
        pnp_id = device.PNPDeviceID
        if name and any(keyword in name.lower() for keyword in CAMERA_KEYWORDS):
            matched_devices.append({
                "name": name,
                "pnp_id": pnp_id
            })

    return matched_devices

if __name__ == "__main__":
    cameras = get_usb_camera_devices()
    for i, cam in enumerate(cameras):
        print(f"{i}: {cam['name']} | PNPDeviceID: {cam['pnp_id']}")
