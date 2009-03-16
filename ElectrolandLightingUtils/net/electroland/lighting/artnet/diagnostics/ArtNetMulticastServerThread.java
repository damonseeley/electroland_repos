package net.electroland.lighting.artnet.diagnostics;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;

import net.electroland.lighting.artnet.ArtNetDMXData;

public class ArtNetMulticastServerThread extends Thread{
	
	long delay;
	int listenPort, sendPort;
	String ipaddress;
	
	public ArtNetMulticastServerThread(String ipaddress, int listenPort, 
										int sendPort, long delay){
		this.delay = delay;
		this.listenPort = listenPort;
		this.sendPort = sendPort;
		this.ipaddress = ipaddress;
	}

	public void run(){

		byte[] onData = new byte[512];
		for (int i=0;i<512;i++) {
			onData[i] = (byte)0xFF;
		}	

		byte[] offData = new byte[512];
		for (int i=0;i<512;i++) {
			offData[i] = (byte)0x00;
		}	
		
		ArtNetDMXData[] seq = new ArtNetDMXData[256];
		for (int even = 0; even < 255; even += 2){
			seq[even] = new ArtNetDMXData();
			seq[even].setPhysical((byte)1);
			seq[even].setUniverse((byte)0);
			seq[even].setData(offData);
			seq[even].Sequence = (byte)(even + 1); // hack.
		}
		for (int odd = 1; odd < 256; odd += 2){
			seq[odd] = new ArtNetDMXData();
			seq[odd].setPhysical((byte)1);
			seq[odd].setUniverse((byte)0);
			seq[odd].setData(onData);			
			seq[odd].Sequence = (byte)(odd); // hack.
		}
		try{
			DatagramSocket socket = new DatagramSocket(listenPort);

			int cnt = 0;
			while (true){
				InetAddress group = InetAddress.getByName(ipaddress);

				// flip between on and off				
				ByteBuffer b = seq[cnt++].getBytes();
				if (cnt == 256){
					cnt = 0;
				}
				DatagramPacket packet 	                
					= new DatagramPacket(b.array(), b.position(), group, sendPort);

				socket.send(packet);

				try{
					sleep(delay);
				}catch(InterruptedException e){
					e.printStackTrace();
				}
			}
		}catch(IOException f){
            f.printStackTrace();				
		}
	}
}