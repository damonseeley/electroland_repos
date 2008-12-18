package net.electroland.detector;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;

import net.electroland.artnet.ip.ArtNetDMXData;

public class ArtNetDMXLightingFixture extends DMXLightingFixture {

	public ArtNetDMXLightingFixture(byte universe, String ipStr, int port,
			int channels, int width, int height, String id) throws UnknownHostException {
		super(universe, ipStr, port, channels, width, height, id);
	}

	@Override
	public void send(byte[] data){
		try {

			ArtNetDMXData dmx = new ArtNetDMXData();
			
			dmx.setUniverse(universe);
			dmx.setPhysical((byte)1);
			dmx.Sequence = (byte)0;	
			dmx.setData(data);

			
			ByteBuffer b = dmx.getBytes();

			
			System.out.println(this.id + "/" + universe + ": " + bytesToHex(b.array(), b.position()));			


			
			// cache these??

			DatagramSocket socket = new DatagramSocket(port);
			DatagramPacket packet = new DatagramPacket(b.array(), b.position(), ip, port);
			socket.send(packet);
			
			socket.close();
			
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static String bytesToHex(byte[] b, int length){
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i< length; i++){
			sb.append(Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ");
		}
		return sb.toString();
	}
}
