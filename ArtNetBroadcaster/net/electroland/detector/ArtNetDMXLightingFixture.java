package net.electroland.detector;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;

import net.electroland.artnet.ip.ArtNetDMXData;

public class ArtNetDMXLightingFixture extends DMXLightingFixture {

	public static int ART_NET_PORT = 6454; // port should be fixed for art net.
	private static DatagramSocket socket;
	
	public ArtNetDMXLightingFixture(String id, byte universe, String ipStr,
			int channels, int width, int height) throws UnknownHostException {
		super(id, universe, ipStr, ArtNetDMXLightingFixture.ART_NET_PORT, channels, width, height);
	}
	
	@Override
	public void send(byte[] data){
		try {

			ArtNetDMXData dmx = new ArtNetDMXData(); // could cache this.
			
			dmx.setUniverse(universe);
			dmx.setPhysical((byte)1);
			dmx.Sequence = (byte)0;	
			dmx.setData(data);

			
			ByteBuffer b = dmx.getBytes();

			if (log){
				System.out.println(this.id + ", universe " + universe + " at IP " + this.ipStr + ":" + bytesToHex(b.array(), b.position()));			
			}

			synchronized (this){
				if (socket == null || socket.isClosed()){
					socket = new DatagramSocket(port);				
				}				
			}

			DatagramPacket packet = new DatagramPacket(b.array(), b.position(), ip, port);
			socket.send(packet);
			
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
