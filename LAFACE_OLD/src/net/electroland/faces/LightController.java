package net.electroland.faces;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.concurrent.CopyOnWriteArrayList;

import net.electroland.artnet.ip.ArtNetDMXData;

/**
 * LightController represents a single recipient of an ArtNet packet.  E.g.,
 * a single lighting universe's data as delivered to a single IP address and
 * port.  Each LightController has connection info, and lighting info.  You
 * set it up by telling it the universe it represents, the ipaddress of the
 * recipient, the port to send to on the recipient, and the port we would 
 * listen to if we cared to.  You add lights to it using the addLight method.
 * 
 * addLight is a little broken right now.  It uses channel as the array index,
 * to solve a non-existent problem.  
 * 
 * If you want to send the state of all your lights to the lighting controller
 * on the end of this, you simply call "send()"
 * 
 * @author geilfuss
 */
public class LightController {

	byte universe;
	int sendPort, listenPort;
	String ipaddress;
	CopyOnWriteArrayList<Light> lights;

	public LightController(byte universe, String ipaddress, int sendPort, int listenPort){
		this.universe = universe;
		this.sendPort = sendPort;
		this.ipaddress = ipaddress;
		this.listenPort = listenPort;
		lights = new CopyOnWriteArrayList<Light>();
	}

	public void addLight(Light l){
		// hack.  we're using channel from the properties file to determine the
		// array index. FUTURE: remove channel from the properties file (and from
		// the Light object) and just put this into an incremental sequence.
		lights.add(l.channel-1,l);
	}
	
	public String toString(){
		StringBuffer b = new StringBuffer("LightController[universe=");
		b.append((int)universe).append(", ipaddress=").append(ipaddress);
		b.append(", port=").append(sendPort).append("]\n");
		for (int i = 0; i < lights.size(); i++){
			b.append('\t').append(lights.get(i)).append('\n');
		}
		return b.toString();
	}
	
	public void allOn(){
		java.util.Iterator<Light> i = lights.iterator();
		while (i.hasNext()){
			Light l = i.next();
			l.color = 255;
			l.brightness = 255;
		}
	}
	
	public void allOff(){
		java.util.Iterator<Light> i = lights.iterator();
		while (i.hasNext()){
			Light l = i.next();
			l.color = 0;
			l.brightness = 0;
		}		
	}
	
	public void send(){
		
		try{
			DatagramSocket socket = new DatagramSocket(listenPort);

			InetAddress group = InetAddress.getByName(ipaddress);

			ArtNetDMXData dmx = new ArtNetDMXData();
			dmx.setUniverse((byte)universe);

			// we don't use these parts of the spec.
			dmx.setPhysical((byte)1);
			dmx.Sequence = (byte)0;	

			
			// set light data
			int length = lights.size();
			byte[] data = new byte[length * 2];
			for (int i=0; i < length; i++){
				Light l = lights.get(i);
				data[i * 2] = (byte)l.color;
				data[(i * 2) + 1] = (byte)l.brightness;				
			}
			dmx.setData(data);

			
			ByteBuffer b = dmx.getBytes();

			//Damon
			if (universe == 0)// note: we're only printing out universe 0.
				System.out.println(universe + ": " + Util.bytesToHex(b.array()));			
			
			DatagramPacket packet 	                
				= new DatagramPacket(b.array(), b.position(), group, sendPort);

			socket.send(packet);
			socket.close();

		}catch(IOException f){
            f.printStackTrace();				
		}
	}
}