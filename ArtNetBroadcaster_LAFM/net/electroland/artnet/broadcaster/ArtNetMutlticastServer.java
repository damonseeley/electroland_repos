package net.electroland.artnet.broadcaster;

public class ArtNetMutlticastServer {

	public static void main(String[] args) throws java.io.IOException {
		
		// no validation right now.
		//  (valid is [xxx.xxx.xxx.xxx] [listen_port] [send_port] [fps])
		//  ( we aren't actually listening to the input port)

		int fps = new Integer(args[3]);
		new ArtNetMulticastServerThread(args[0], 
										new Integer(args[1]).intValue(),
										new Integer(args[2]).intValue(), 
										(long)(1000 / fps)).start();
	}	
}