package net.electrolnd.installutils.mgmt;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class ProcessClient {

	public static void main(String args[])
	{
		int port = 8181; // default port.
		ServerSocket me = null;

		try 
		{
			if (args.length == 1)
				port = Integer.parseInt(args[0]);

			me = new ServerSocket(port);
			
		} catch (IOException e) {
		    System.out.println("Could not listen on port: " + port);
		    System.exit(-1);
		} catch (NumberFormatException e){
		    System.out.println("Invalid port: " + args[0]);			
		    System.exit(-1);
		}

		try 
		{
			Process process = null;
			Socket master = me.accept();

			System.out.println("Accepting connection from " + master.getInetAddress());

			while (true)
			{
				BufferedReader in = new BufferedReader(
	                      new InputStreamReader(
	                    		  master.getInputStream()));

				String command = in.readLine();

				if ("start".equalsIgnoreCase(command))
				{
					if (process != null)
					{
						process.destroy();
						process.waitFor();
						System.out.println("killed existing process.");
					}
					process = Runtime.getRuntime().exec("ls -al");
					System.out.println("started new process");
					
				}else if ("stop".equalsIgnoreCase(command))
				{
					if (process != null)
					{
						process.destroy();
						process.waitFor();
						System.out.println("killed existing process.");
					}					
				}
			}
			
		} catch (IOException e) {
			System.out.println("Connection failure: " + e);
		    System.exit(-1);
		} catch (InterruptedException e) {
			System.out.println("InterruptedException: " + e);
			e.printStackTrace();
		}
	}
}