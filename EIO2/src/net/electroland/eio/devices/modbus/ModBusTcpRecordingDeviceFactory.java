package net.electroland.eio.devices.modbus;

import net.electroland.eio.Device;
import net.electroland.eio.DeviceFactory;
import net.electroland.utils.ParameterMap;

public class ModBusTcpRecordingDeviceFactory extends DeviceFactory {

    @Override
    public Device create(ParameterMap deviceParams) {
        return new ModBusTcpRecordingDevice(deviceParams);
    }
}