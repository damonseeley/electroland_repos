package net.electroland.eio.devices;

import net.electroland.eio.IOState;

abstract public class IODevice {

	abstract public void connect();

	abstract public void sendOutput();

	abstract public void readInput();

	abstract public void close();

	abstract public void patch(IOState state, String port);
}