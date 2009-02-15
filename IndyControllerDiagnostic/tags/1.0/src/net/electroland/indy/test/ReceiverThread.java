package net.electroland.indy.test;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

public class ReceiverThread extends Thread {

	Target target;
	
	public ReceiverThread(Target target){
		this.target = target;
	}
	
	@Override
	public void run() {

//		int ctr = 0;
		DatagramSocket socket;

		try {
			System.out.println("listening on " + target.getPort());
			System.out.println("--");
			socket = new DatagramSocket(target.getPort());

			while(true){
				DatagramPacket p = new DatagramPacket(new byte[15], 15);
				socket.receive(p);
				byte[] receivedData = p.getData();
/* for output the time
				long receiveTime = System.currentTimeMillis();
				byte[] sendTimeBytes = new byte[6];
				System.arraycopy(receivedData, 8, sendTimeBytes, 0, 6);
				long sendTime = new BigInteger(sendTimeBytes).longValue();
				long roundTrip = receiveTime - sendTime;

				if (ctr++ % 100 == 0){
					System.out.println("Packet on port " + target.getPort() + " took " + roundTrip + " milliseconds round trip.");
				}

*/				
				System.out.println(target + " Packet returned= " + Util.bytesToHex(receivedData));
			}
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}