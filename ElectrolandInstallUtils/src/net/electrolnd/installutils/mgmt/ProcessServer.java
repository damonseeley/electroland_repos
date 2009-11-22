package net.electrolnd.installutils.mgmt;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Properties;

public class ProcessServer {

	public static void main(String args[])
	{
		try {
			// load props
			Properties p = new Properties();
			p.load(new FileInputStream(new File("clients.properties")));

			Enumeration <Object>keys = p.keys();
			ArrayList <String> clientStrs = new ArrayList<String>();

			// expecting:
			//	client0=127.0.0.1:8181
			//	client0=127.0.0.2:8181
			//	etc.
			while (keys.hasMoreElements()){

				Object key = keys.nextElement();

				if (key instanceof String && 
					((String)key).startsWith("client"))
				{
					clientStrs.add(p.getProperty((String)key));					
				}
			}		

			// get the sockets
			ArrayList <Connection>connections = parse(clientStrs);
			ArrayList <ClientJFrame> clients = new ArrayList<ClientJFrame>();
			Iterator <Connection>i = connections.iterator();
			while (i.hasNext())
			{
				ClientJFrame client = new ClientJFrame(i.next());
				clients.add(client);
				new Thread(client).start();
			}
			new MasterJFrame(clients);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static ArrayList <Connection> parse(ArrayList <String> clientsStr) throws IOException
	{
		ArrayList <Connection> connections = new ArrayList<Connection>();
		Iterator <String> i = clientsStr.iterator();
		while (i.hasNext())
		{
			String str = i.next();
			int portMark = str.indexOf(':');
			if (portMark != -1)
			{
				String ip = str.substring(0, portMark);
				String portStr = str.substring(portMark + 1, str.length());
				try
				{
					int port = Integer.parseInt(portStr);
					connections.add(new Connection(ip, port));
					
				}catch(NumberFormatException e)
				{
					throw new IOException("bad client (bad port specified): " + str);					
				}
			}else
			{
				throw new IOException("bad client (no port specified): " + str);
			}
		}
		return connections;
	}
}