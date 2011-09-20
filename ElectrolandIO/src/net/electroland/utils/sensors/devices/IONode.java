package net.electroland.utils.sensors.devices;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import net.electroland.utils.sensors.IOState;

// TODO: rename this to IONode
abstract public class IONode {

	abstract public void connect();

	/**
	 * Needs to read all the OStates and send their data to the proper ports.
	 */
	abstract public void sendOutput();

	/**
	 * Needs to read data, then map and write the data to the IStates
	 */
	abstract public void readInput();

	abstract public void close();

	private Map<String,IOState> states = Collections.synchronizedMap(new HashMap<String,IOState>());

	/**
	 * Concrete patching (this isn't quite right: it needs to translate
	 * 
	 * phoenix4DI8DO.register1 = $startRef 0
	 * phoenix4DI8DO.patch0 = $register register1 $bit 8 $dinput 0
	 * iostate.0 = $ionode phoenix1 $dinput 0 $tags "A ALL" $x 0 $y 0 $z 0 $units meters $filters net.electroland.eio.filters.BitLoPass
	 * 
	 * To a descrete patch.  Concrete instances of this wil have received the first two lines
	 * AND the $ionode phoenix1 $dinput 0 portion of the second line.
	 * 
	 * The IOState will have received the rest of the second line.
	 * 
	 * Therefore, the patch process has to do a lookup for $dinput 0, and patch the IOState to it.
	 * 
	 * After that, sendOutput and readOutput should be able to easily map input data to each state.
	 * 
	 * @param state
	 */
	final public void patch(IOState state, String port)
	{
		states.put(port, state);
	}
}