package net.electroland.memphis.utils.bytelistener;

import java.net.SocketException;
import java.net.UnknownHostException;

public class ByteListenerMain {
	
	UDPParser parser;
	
	public ByteListenerMain(int port){
		try {
			parser = new UDPParser(port);
			parser.run();
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		int port = 1001;
		if(args.length > 0){
			port = Integer.parseInt(args[0]);
		}
		ByteListenerMain main = new ByteListenerMain(port);
		
	}

}
