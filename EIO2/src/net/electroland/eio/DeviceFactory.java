package net.electroland.eio;

import net.electroland.utils.ParameterMap;

public abstract class DeviceFactory {
    abstract public Device create(ParameterMap deviceParams);
}