using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace Electroland
{
	public class UDPSender
	{
			IPEndPoint broadcastIp;
			UdpClient sender;
			
		public UDPSender (int port)
		{
			broadcastIp = new IPEndPoint (IPAddress.Broadcast, port);
			sender = new UdpClient ();
			sender.EnableBroadcast = true;
		}
		
		public void send(String msg) {
			byte[] data = Encoding.ASCII.GetBytes (msg);
			sender.BeginSend (data, data.Length, broadcastIp, new AsyncCallback(SendCallback), sender);
		}
		
		public static void SendCallback(IAsyncResult ar) {
  			UdpClient u = (UdpClient)ar.AsyncState;
			u.EndSend(ar);
		}
		
		public void close() {
			sender.Close();
		}
	}

}
