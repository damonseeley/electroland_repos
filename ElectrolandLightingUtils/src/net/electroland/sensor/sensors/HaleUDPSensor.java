package net.electroland.sensor.sensors;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

import net.electroland.sensor.Sensor;
import net.electroland.sensor.events.HaleUDPSensorEvent;

import org.apache.log4j.Logger;

public class HaleUDPSensor extends Sensor implements Runnable {

	private static Logger logger = Logger.getLogger(HaleUDPSensor.class);
		
	private int port, maxPacketSize;
	private boolean isRunning = false;
	private Thread thread = null;

	public HaleUDPSensor(int port, int maxPacketSize)
	{
		this.port = port;
		this.maxPacketSize = maxPacketSize;
	}

	// for testing
	public static void main(String args[])
	{
		new HaleUDPSensor(7474, 2048).startSensing();
	}
	

	@Override
	public void startSensing() {
		isRunning = true;
		thread = new Thread(this);
		thread.start();
	}

	@Override
	public void stopSensing() {
		isRunning = false;
	}

	@Override
	public void run() {

		DatagramSocket socket;
		try {
			socket = new DatagramSocket(port);

			byte[] buffer = new byte[maxPacketSize];
			DatagramPacket packet = new DatagramPacket(buffer, buffer.length);

			while (isRunning) {
				socket.receive(packet);
				HaleUDPSensorEvent event 
					= new HaleUDPSensorEvent(packet.getAddress().getHostName(), packet.getData());
				if (event.isValid()){
					this.notifyListeners(event);
					logger.debug(event);
				}else{
					logger.error("Invalid HaleUDPSensorEvent: " + event);
				}
				packet.setLength(buffer.length);
			}

		} catch (SocketException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}finally
		{
			isRunning = false;
			thread = null;
		}
	}
}