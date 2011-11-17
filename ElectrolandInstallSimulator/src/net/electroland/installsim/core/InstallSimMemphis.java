package net.electroland.installsim.core;

//import javax.swing.JApplet;

public class InstallSimMemphis extends Thread {

	/*
	 * deleted other gibberish.  main difference here is that memphis uses HaleUDP for now and must include this code and remove SCSC specific code
	 */
	
	public static String address = "localhost";
	public static int port = 7474;
	public static HaleUDPoutput hudp = new HaleUDPoutput(address, port);
	
	public InstallSimMemphis() {
		

	}

	

	
	public static void main(String[] args) {
		new InstallSimMemphis();
	}
	
	
}
