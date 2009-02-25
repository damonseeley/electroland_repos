package net.electroland.detector;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;

public class HaleUDPLightingFixture extends DMXLightingFixture {

	private static DatagramSocket socket;

	public HaleUDPLightingFixture(String id, byte universe,
			String ipStr, int port, int channels, int width, int height)
			throws UnknownHostException {
		super(id, universe, ipStr, port, channels, width, height);
	}

	@Override
	void send(byte[] data) {

		byte[] protocolAndData = new byte[data.length + 3];

		System.arraycopy(data, 0, protocolAndData, 2, data.length);
		
		protocolAndData[0] = (byte)0;
		protocolAndData[1] = (byte)0;
		protocolAndData[protocolAndData.length - 1] = (byte)255;

		if (log)
		{
			System.out.println(this.id + ", universe " + universe + " at IP " + 
								this.ipStr + ":" + 
								bytesToHex(protocolAndData, protocolAndData.length));			
		}		
		
		synchronized (this)
		{
			if (socket == null || socket.isClosed())
			{
				try 
				{
					socket = new DatagramSocket(port);
					DatagramPacket packet = new DatagramPacket(protocolAndData, 
										protocolAndData.length, ip, port);
					socket.send(packet);
				} catch (SocketException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}				
			}				
		}		
	}
}