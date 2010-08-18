package net.electroland.installsim.core;
import java.io.*;
import java.net.*;

public class HaleUDPoutput {
	
	public boolean ready;
	public InetAddress address;
	public int port;
	public DatagramSocket socket;
	
	//constructor
	public HaleUDPoutput(String theAddress, int thePort){
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

        // send packet
        byte[] buf = new byte[30];
        buf = thePacket.getBytes();
        buf = HexUtils.hexToBytes(thePacket);
	    //System.out.println(HexUtils.hexToBytes(thePacket).toString());
        DatagramPacket packet = new DatagramPacket(buf, buf.length, address, port);
        
        try {
			socket.send(packet);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

	public void closeSocket() {
        socket.close();
	}
    
    
    
    
}
