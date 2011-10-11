package net.electroland.eio.devices;

import java.util.Map;

import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

abstract public class IONodeFactory {

    /**
     * Configure the Factory using all variables assigned to this device.
     * 
     * e.g:
     * 
     * if we are creating a device factory for:
     * ionodeType.phoenix4DI8DO = $factory net.electroland.eio.devices.PhoenixILETHBKFactory
     * 
     * pass all these:
     * phoenix4DI8DO.register1 = $startRef 0
     * phoenix4DI8DO.patch0 = $register register1 $bit 8 $port 0
     * phoenix4DI8DO.patch1 = $register register1 $bit 9 $port 1
     * 
     * 
     * This needs to generate the port map
     * 
     * @param names
     * @param props
     */
    abstract public void prototypeDevice(Map<String, ParameterMap> params) throws OptionException;

    /**
     * create an instance of this device, using the prototype and the configuration
     * params for this instance.
     * 
     * e.g.: ionode.phoenix1 = $type phoenix4DI8D0 $ipaddress 192.168.1.61
     * 
     * @param paramNames
     * @param props
     * @return
     */
    abstract public IONode createInstance(ParameterMap params) throws OptionException;
}
