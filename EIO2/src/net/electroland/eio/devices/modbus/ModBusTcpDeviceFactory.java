package net.electroland.eio.devices.modbus;

import net.electroland.eio.devices.Device;
import net.electroland.eio.devices.DeviceFactory;
import net.electroland.utils.ParameterMap;


public class ModBusTcpDeviceFactory extends DeviceFactory {

    @Override
    public Device create(ParameterMap deviceParams) {
        return new ModBusTcpDevice(deviceParams);
    }
}