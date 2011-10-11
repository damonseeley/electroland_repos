package net.electroland.eio.devices;

import net.electroland.eio.IOState;

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

	abstract public void patch(IOState state, String port);
}