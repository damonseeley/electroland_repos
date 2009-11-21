package net.electrolnd.installutils.mgmt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Properties;

public class ProcessClient {

//	final static String startCmd = "C:\\Documents and Settings\\Electroland\\Desktop\\FIL\\Projection2\\FILMain";
//	final static String stopCmd = "taskkill /f /fi \"WINDOWTITLE eq FIL*\"";
	
	public static void main(String args[])
	{
		try 
		{
			// load props
			Properties p = new Properties();
			p.load(new FileInputStream(new File("directives.properties")));

			// command to be run on 'start' requests
			String startCmd = p.getProperty("startDirective");
			// command to be run on 'stop' requests (and before 'start')
			String stopCmd = p.getProperty("stopDirective");
			// port to listen on.
			String portStr = p.getProperty("port", "8181");

			int port = Integer.parseInt(portStr);
			ServerSocket me = new ServerSocket(port);
			System.out.println("Waiting for Director connection...");

			while (true)
			{
				try 
				{
					Socket master = me.accept();
					System.out.println("Director connected from " + master.getInetAddress());

					PrintWriter out = new PrintWriter(
		                      new OutputStreamWriter(
		                    		  master.getOutputStream()));					
					out.println("Welcome Director!");
					out.println("=================");
					out.println(" Valid commands are:");
					out.println("   start");
					out.println("   stop");
					out.println("   quit (or just 'q')");
					out.println("");
					out.flush();

					BufferedReader in = new BufferedReader(
		                      new InputStreamReader(
		                    		  master.getInputStream()));
					while (true)
					{
						String directive = in.readLine();

						if (directive.equalsIgnoreCase("start"))
						{
							runCmd(stopCmd,  ">> services stopped.", out);
							runCmd(startCmd, ">> services started.", out);
		
						}else if (directive.equalsIgnoreCase("stop"))
						{
							runCmd(stopCmd, ">> services stopped.", out);

						}else if (directive.toLowerCase().startsWith("q"))
						{
							System.out.println("Director disconnected.");
							out.println(">> good bye!.");
							out.flush();
							master.close();
							break;
						}else {
							System.out.println(">> unknown directive: " + directive);							
							out.println(">> unknown directive: " + directive);
							out.flush();
						}
					}
					
				} catch (IOException e) {
					e.printStackTrace();
				}			
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		    System.exit(-1);
		} catch (NumberFormatException e){
			e.printStackTrace();
		    System.exit(-1);
		}
	}

	private static Process runCmd(String cmd, String successMsg, PrintWriter out)
	{
		System.out.println("issuing command:");
		System.out.println("\t" + cmd);
		System.out.println("");

		try{			
			Process p = Runtime.getRuntime().exec(cmd);
			out.println(successMsg);
			out.flush();
			return p;
			
		}catch (IOException e){
			e.printStackTrace();
			e.printStackTrace(out);
			out.flush();
			return null;
		}
	}
}