package net.electroland.coopLights.core;
import java.io.*;
import java.net.*;

public class UDPSoundComm {
	
	public boolean ready;
	public InetAddress address;
	public int port;
	public DatagramSocket socket;
	
	//constructor
	public UDPSoundComm(String theAddress, int thePort){
		try {
            // get a datagram socket
			address = InetAddress.getByName(theAddress);
			port = thePort;
			socket = new DatagramSocket();
			
			ready = true;
		} catch (Exception e) {
			// TODO: handle exception
			ready = false;
		}		
	}
	
	public void sendPacket (String thePacket) {

        // send request
        byte[] buf = new byte[256];
        buf = thePacket.getBytes();
        DatagramPacket packet = new DatagramPacket(buf, buf.length, address, port);
        
        try {
			socket.send(packet);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
	
	public void triggerAmbient(int soundCode){
		this.sendPacket("0 " + soundCode);
	}
	
	public void triggerAvatar(int soundCode, String gains) {
		//System.out.println("1 "+ soundCode + " " + gains);
		this.sendPacket("1 "+ soundCode + " " + gains);
	}
	
	public void closeSocket() {
        socket.close();
	}
    
    
    
    
}
