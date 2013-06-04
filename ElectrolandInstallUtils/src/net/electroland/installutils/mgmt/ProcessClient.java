package net.electroland.installutils.mgmt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Properties;

public class ProcessClient {

    public static final String START_CMD = "start";
    public static final String STOP_CMD = "stop";
    public static final String QUIT_CMD = "quit";
    
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
                    out.println("   " + START_CMD + " [args]");
                    out.println("   " + STOP_CMD);
                    out.println("   " + QUIT_CMD);
                    out.println("");
                    out.flush();

                    BufferedReader in = new BufferedReader(
                              new InputStreamReader(
                                      master.getInputStream()));
                    while (true)
                    {
                        String directive = in.readLine();
                        if (directive == null)
                        {
                            System.out.println("connection broken.");
                            break;
                        }
                        if (directive.toLowerCase().startsWith((START_CMD)))
                        {
                            runCmd(stopCmd,  ">> services stopped.", out);
                            String userArgs = (directive.length() > 5) ?
                                    directive.substring(5, directive.length()) :
                                    "";
                            runCmd(startCmd + userArgs, ">> services started.", out);
        
                        }else if (STOP_CMD.equalsIgnoreCase(directive))
                        {
                            runCmd(stopCmd, ">> services stopped.", out);

                        }else if (QUIT_CMD.equalsIgnoreCase(directive))
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
                    me.close();
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