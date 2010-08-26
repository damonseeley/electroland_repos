package net.electroland.input.devices.memphis;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

import net.electroland.input.InputDevice;
import net.electroland.input.events.HaleUDPInputDeviceEvent;

import org.apache.log4j.Logger;

public class HaleUDPInputDevice extends InputDevice implements Runnable {

	private static Logger logger = Logger.getLogger(HaleUDPInputDevice.class);
		
	private int port, maxPacketSize;
	private boolean isRunning = false;
	private Thread thread = null;

	public HaleUDPInputDevice(int port, int maxPacketSize)
	{
		this.port = port;
		this.maxPacketSize = maxPacketSize;
	}

	// for testing
	public static void main(String args[])
	{
		new HaleUDPInputDevice(7474, 2048).startSensing();
	}

	public void startSensing() {
		isRunning = true;
		thread = new Thread(this);
		thread.start();
	}

	public void stopSensing() {
		isRunning = false;

		if (socket != null){
			socket.close();
		}

	}
	private DatagramSocket socket = null;

	public void run() {

		try {
			
			
			socket = new DatagramSocket(port);

			byte[] buffer = new byte[maxPacketSize];
			DatagramPacket packet = new DatagramPacket(buffer, buffer.length);

			while (isRunning) {
				socket.receive(packet);
				HaleUDPInputDeviceEvent event 
					= new HaleUDPInputDeviceEvent(packet.getAddress().getHostName(), packet.getData());
				if (event.isValid()){
					this.notifyListeners(event);
					logger.debug(event);
				}else{
					logger.error("Invalid HaleUDPSensorEvent: " + event);
				}
				packet.setLength(buffer.length);
			}
			
		} catch (SocketException e) {
			if (isRunning){
				logger.error(e);
			}
		} catch (IOException e) {
			logger.error(e);
		}finally
		{
			isRunning = false;
			thread = null;
		}
	}
}