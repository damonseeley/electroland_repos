package net.electroland.indy.test;

import java.net.UnknownHostException;

/**
 * 
 * @author geilfuss
 *
 *	This is a quick and dirty loop back test.  You specify the ipaddress:port
 *  addresses that you want to send packets too.  It sends, and then checks
 *  for packets returned.  The 8-14th bytes contain the send time, and so the
 *  receivers just determine the difference between send and receive.
 *
 */
public class TestRunner {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		try{
			// arg[0] = the fps
			int fps = Integer.parseInt(args[0]);

			// rest of the args are ipaddres:port
			Target[] targets = new Target[args.length - 1];
			for (int i = 1; i < args.length; i++){
				targets[i - 1] = new Target(args[i]);
			}

			// create a sender
			new SenderThread(fps, targets).start();	

			// create receiver threads
			for (int i=0; i < targets.length; i++ ){
				new ReceiverThread(targets[i]).start();
			}			

		}catch(NumberFormatException e){
			e.printStackTrace();			
		}catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (IPAddressParseException e) {
			e.printStackTrace();
		}finally{
			System.out.println("usage: java TestRunner [fps] [host1:port1]...");
		}
	}
}