using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;

namespace Electroland
{
	
	public class UDPReceiver
	{
		public class UdpState {
		public IPEndPoint e ;
		public UdpClient u ;
	}

		ThreadedStringListener listener;
		UdpClient udpClient;
		protected Thread receiveLoopThread;
		UdpState state = new UdpState();
		
		public UDPReceiver (int port, ThreadedStringListener l)
		{
			listener = l;
			listener.start();
			IPEndPoint ipEndPoint = new IPEndPoint(IPAddress.Any, port);
			udpClient = new UdpClient(ipEndPoint);
			state.e = ipEndPoint;
			state.u = udpClient;
			udpClient.BeginReceive(new AsyncCallback(receive), state);
			
		}
		
		
		public void receive(IAsyncResult ar) {
			 UdpClient u = (UdpClient)((UdpState)(ar.AsyncState)).u;
			IPEndPoint e = (IPEndPoint)((UdpState)(ar.AsyncState)).e;
			
			
			Byte[] receiveBytes = u.EndReceive(ar, ref e);
			
			
			string receiveString = Encoding.ASCII.GetString(receiveBytes);
//			Console.WriteLine("Receiving " + receiveString);
			
			listener.put(receiveString);
			udpClient.BeginReceive(new AsyncCallback(receive), state);
		}
		
		
		public void close() {
			udpClient.Close();
			listener.end();
		}
		
		
	}
}

