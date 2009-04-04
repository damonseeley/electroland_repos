package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;

import net.electroland.lighting.artnet.ArtNetDMXData;

public class ArtNetRecipient extends Recipient {

	protected byte universe;
	public static int ART_NET_PORT = 6454; // port should be fixed for art net.
	private static DatagramSocket socket;

	public ArtNetRecipient(String id, byte universe, String ipStr,
			int channels, Dimension preferredDimensions) throws UnknownHostException {
		super(id, ipStr, ArtNetRecipient.ART_NET_PORT, channels, preferredDimensions);
		this.universe = universe;
	}

	public ArtNetRecipient(String id, byte universe, String ipStr,
			int channels, Dimension preferredDimensions, String patchgroup) throws UnknownHostException {
		super(id, ipStr, ArtNetRecipient.ART_NET_PORT, channels, preferredDimensions, patchgroup);
		this.universe = universe;
	}

	final public byte getUniverse() {
		return universe;
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
