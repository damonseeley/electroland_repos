package net.electroland.installutils.mgmt;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

public class Connection {

	protected String address;
	protected int port;
	protected Socket socket;
	protected BufferedReader responseStream;

	public Connection(String address, int port){
		this.address = address;
		this.port = port;
	}

	public void connect() throws UnknownHostException, IOException
	{
		if (socket == null || !socket.isConnected())
		{
			socket = new Socket(address, port);			
			responseStream = new BufferedReader(new InputStreamReader(socket.getInputStream()));
		}
	}

	public void sendCommand(String command) throws UnknownHostException, IOException{

		connect();

		PrintWriter pw = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()));
		pw.println(command);
		pw.flush();
	}
	
	public void start() throws UnknownHostException, IOException{
		sendCommand(ProcessClient.START_CMD);
	}

	public void stop() throws UnknownHostException, IOException{
		sendCommand(ProcessClient.STOP_CMD);
	}	
}