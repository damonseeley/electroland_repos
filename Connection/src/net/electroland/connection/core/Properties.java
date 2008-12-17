package net.electroland.connection.core;

import java.net.InetAddress;
import java.io.*;

/**
 * 
 * This class acts as an object to hold all of the systems properties global variables,
 * as well as load them from the start.
 * @author Aaron Siegel
 *
 */

public class Properties {
	
	InetAddress address;
	String ip;
	BufferedReader input;
	String line;

	public Properties(){
		loadProperties();
	}
	
	public void loadProperties(){
		try{
			input = new BufferedReader(new FileReader("depends/properties.conf"));
		} catch (FileNotFoundException e){
			e.printStackTrace();
		}
		try{
			while((line = input.readLine()) != null){
				System.out.println("'"+line.split("=")[0].trim()+"'");
				if(line.split("=")[0].trim() == "Subnet"){
					System.out.println("yes");
				}
			}
		} catch (IOException e){
			e.printStackTrace();
		}
	}
	
}
