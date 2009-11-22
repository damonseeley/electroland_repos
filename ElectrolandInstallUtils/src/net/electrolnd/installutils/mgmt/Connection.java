package net.electrolnd.installutils.mgmt;

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

	public Connection(String address, int port){
		this.address = address;
		this.port = port;
	}

	public String sendCommand(String command)throws UnknownHostException, IOException{

		if (socket == null || !socket.isConnected())
		{
			socket = new Socket(address, port);
		}

		BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream()));
		PrintWriter pw = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()));

		pw.println(command);

		StringBuffer sbr = new StringBuffer();
		while (br.ready()){
			sbr.append(br.readLine()).append("\n\r");
		}
		return sbr.toString();
	}
	
	public String start()throws UnknownHostException, IOException{
		return sendCommand(ProcessClient.START_CMD);
	}

	public String stop()throws UnknownHostException, IOException{
		return sendCommand(ProcessClient.STOP_CMD);
	}	
}