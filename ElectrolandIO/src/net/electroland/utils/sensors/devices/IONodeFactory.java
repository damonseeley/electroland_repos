package net.electroland.utils.sensors.devices;

import java.util.Map;

// TODO: rename this to IONodeFactory
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
	 * ...
	 * 
	 * This needs to generate the port map
	 * 
	 * @param config
	 * @return
	 */
	abstract public void prototypeDevice(Map<String, Map<String,String>> config);

	/**
	 * create an instance of this device, using the prototype and the configuration
	 * params for this instance.
	 * 
	 * e.g.: ionode.phoenix1 = $type phoenix4DI8D0 $ipaddress 192.168.1.61
	 * 
	 * @param config
	 * @return
	 */
	abstract public IONode [] createInstance(Map<String,String> config);
}
