package net.electroland.eio.devices.modbus;

import net.electroland.eio.Device;
import net.electroland.eio.DeviceFactory;
import net.electroland.utils.ParameterMap;

public class ModBusTcpPlaybackDeviceFactory extends DeviceFactory {

    @Override
    public Device create(ParameterMap deviceParams) {
        return new ModBusTcpPlaybackDevice(deviceParams);
    }
}