package net.electroland.memphis.utils.bytelistener;

import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class ByteListenerMain {
	
	UDPParser parser;
	UDPLogger logger;
	
	public ByteListenerMain(int port){
		try {
			System.out.println("Attempting bind on port " + port);
			parser = new UDPParser(port);
			parser.run();
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
	}
	
	public ByteListenerMain(int port, String logFileName){
		try {
			System.out.println("Started up with filename " + logFileName);
			logger = new UDPLogger(logFileName, port);
			logger.run();
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		//System.out.println(args[0] + " : " + args[1]);
		int port;
		String fileName;
		if(args.length == 1){
			port = Integer.parseInt(args[0]);
			ByteListenerMain main = new ByteListenerMain(port);
		} else if(args.length > 1) {
			Date dateNow = new Date ();
	        SimpleDateFormat dateformatYYYYMMDD = new SimpleDateFormat("E_yyyy_MM_dd-HHmmss_a"); 
	        StringBuilder nowYYYYMMDD = new StringBuilder( dateformatYYYYMMDD.format( dateNow ) );

			port = Integer.parseInt(args[0]);
			fileName = args[1];
			ByteListenerMain main = new ByteListenerMain(port, fileName+"_"+nowYYYYMMDD+".txt");
		}
		
		
	}

}
