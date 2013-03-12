package net.electroland.eio.devices;

import net.electroland.utils.ParameterMap;

public abstract class DeviceFactory {
    abstract public Device create(ParameterMap deviceParams);
}