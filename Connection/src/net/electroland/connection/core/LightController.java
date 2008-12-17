package net.electroland.connection.core;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.net.InetAddress;

/**
 * Main control over all light activity.
 */

public class LightController {
	
	DatagramSocket socket;						// socket to lights
	InetAddress address;						// broadcast address
	//public Collection <Person> peopleCollection;
	//public Collection <Link> linkCollection;
	public LightBox[] lightboxes;				// light controllers
	//public Light[] lights;						// grid of lights
	public byte[] buffer, newbuffer, oldbuffer;// raw light data
	public byte[] pair;						// pair of red and blue bytes for a single light
	public static int gridx;					// grid dimensions
	public static int gridy;
	public String bytestring;					
	public int displayMode, lastMode;			// current mode and previous mode
	//public Conductor conductor;				// controls animation transitions

	public LightController(int gx, int gy){
		gridx = gx;
		gridy = gy;
		displayMode = 0;
		pair = new byte[2];
		try{
			socket = new DatagramSocket();					// sends data to lights
		} catch (SocketException e) {
			e.printStackTrace();
		}
		try{
			address = InetAddress.getByName(ConnectionMain.properties.get("Subnet")+"."+ConnectionMain.properties.get("BroadcastTarget"));
			System.out.println("broadcast address: "+ ConnectionMain.properties.get("Subnet")+"."+ConnectionMain.properties.get("BroadcastTarget"));
		} catch(UnknownHostException e){
			e.printStackTrace();
		}
		lightboxes = new LightBox[14];						// number of lighting controllers
		setupLightBoxes();
		/*
		lights = new Light[gridx*gridy];					// empty array
		int count = 0;										// count x and y
		for(int y=0; y<gridy; y++){						// for each y position
			for(int x = 0; x<gridx; x++){					// for each x position
				lights[count] = new Light(count, x, y);		// create a new light
				count++;
			}
		}
		*/
		//conductor = new Conductor(lights);
		//peopleCollection = ConnectionMain.people.values();
		//linkCollection = ConnectionMain.links.values();
	}
	
	public void setupLightBoxes(){
		for(int i=0; i<lightboxes.length; i++){
			System.out.println("lightbox #"+i+": "+ ConnectionMain.properties.get("Subnet")+"."+(21+i));
			lightboxes[i] = new LightBox(i, ConnectionMain.properties.get("Subnet")+"."+(21+i), 24*i);
			//lightboxes[i] = new LightBox(i, "192.168.1."+(21+i), 24*i);	// SUBNET AND STARTING IP FOR LIGHT CONTROLLERS!
		}
	}
	
	public void updateLightBoxes(byte[] buffer){
		// FF 01 00 00 00 00 00 00 00 00 00 00 00 00 FF 80 00 24 FE
		// start byte , command byte, 12 bytes of data, start byte, offset byte, 2 bytes offset value, end byte 
		byte[] datachunkA = new byte[19];
		datachunkA[0] = (byte)255;
		datachunkA[1] = (byte)1;
		datachunkA[14] = (byte)255;
		datachunkA[15] = (byte)128;
		datachunkA[18] = (byte)254;
		byte[] datachunkB = new byte[19];
		datachunkB[0] = (byte)255;
		datachunkB[1] = (byte)1;
		datachunkB[14] = (byte)255;
		datachunkB[15] = (byte)128;
		datachunkB[18] = (byte)254;
		for(int i=0; i<lightboxes.length; i++){					// unicast update + data to each lightbox
			System.arraycopy(buffer, i*24+2, datachunkA, 2, 12);	// copy chunk of light data for that controller
			System.arraycopy(buffer, i*24+2+12, datachunkB, 2, 12);
			if(lightboxes[i].offsetA > 253){
				datachunkA[16] = (byte)(lightboxes[i].offsetA-253);
				datachunkA[17] = (byte)253;
			} else {
				datachunkA[16] = (byte)0;
				datachunkA[16] = (byte)lightboxes[i].offsetA;
			}
			if(lightboxes[i].offsetB > 253){
				datachunkB[16] = (byte)(lightboxes[i].offsetB-253);
				datachunkB[17] = (byte)253;
			} else {
				datachunkB[16] = (byte)0;
				datachunkB[16] = (byte)lightboxes[i].offsetB;
			}
			DatagramPacket command = new DatagramPacket(datachunkA, datachunkA.length, lightboxes[i].ip, Integer.parseInt(ConnectionMain.properties.get("SendPortA")));	// new packet
			
			try {
				socket.send(command);								// send command
			} catch (IOException e) {
				System.out.println(e);
			}
			command = new DatagramPacket(datachunkB, datachunkB.length, lightboxes[i].ip, Integer.parseInt(ConnectionMain.properties.get("SendPortB")));	// new packet
			try {
				socket.send(command);								// send command
			} catch (IOException e) {
				System.out.println(e);
			}			
		}
	}
	
	public void updateLights(byte[] buffer){
		//buffer = conductor.draw();	// controls all transitions between display modes
		rampOutput(buffer);							// ramps the byte values to compensate for PWM
		
		// process byte string
		DatagramPacket packet1 = new DatagramPacket(buffer, buffer.length, address, Integer.parseInt(ConnectionMain.properties.get("SendPortA")));	// new packet
		DatagramPacket packet2 = new DatagramPacket(buffer, buffer.length, address, Integer.parseInt(ConnectionMain.properties.get("SendPortB")));	// new packet
		try {
			socket.send(packet1);							// send data
			socket.send(packet2);							// send data
		} catch (IOException e) {
			System.out.println(e);
		}
	}
	
	public void rampOutput(byte[] buffer){				// compensate logarithmically for PWM ramping
		for(int i=2; i<buffer.length-1; i++){				// for each color value in each light...
			float x = ((float)(buffer[i] & 0xFF)/255);
			float y = (float)(Math.pow(1.1, 25.2*x) - 1) / 10;
			buffer[i] = (byte)(int)(y*255);
			//  System.out.println(temp);
		}
	}
	
	public void sendKillPackets(){
		buffer = new byte[ConnectionMain.renderThread.lights.length*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		
		for(int i=2; i<buffer.length-1; i++){				// full of 0 values to turn lights off
			buffer[i] = (byte)0;
		}
		
		for(int i=0; i<4; i++){							// send 4 kill packets
			DatagramPacket packet1 = new DatagramPacket(buffer, buffer.length, address, Integer.parseInt(ConnectionMain.properties.get("SendPortA")));	// new packet
			DatagramPacket packet2 = new DatagramPacket(buffer, buffer.length, address, Integer.parseInt(ConnectionMain.properties.get("SendPortB")));	// new packet
			try {
				socket.send(packet1);							// send data
				socket.send(packet2);							// send data
				//System.out.println(packet1.getLength());
			} catch (IOException e) {
				System.out.println(e);
			}
		}
		//System.out.println("kill kill kill");
	}
}
