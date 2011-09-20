package net.electroland.utils.sensors.devices;

import java.util.Map;

public class PhoenixIONodeFactory extends IONodeFactory {

	@Override
	public void prototypeDevice(Map<String, Map<String, String>> config) {
		// TODO Auto-generated method stub
		// get registers (name starts with register: kludgy)
		// for each register
		//   store the register name and start bit
		// get patches
		// for each patch
		//   find the register, store the register + bit hashed by port
	}

	@Override
	public IONode[] createInstance(Map<String, String> config) {
		
		// create an instance of IODevice
		// give it the port map
		// give it the IP address
		return null;
	}
}