using UnityEngine;
using System.Collections;
using System.Net.Sockets;
using System.Net;
using System.Text;

public class Server : MonoBehaviour {
	
    private Socket socket;					// socket server is binding to
	private EndPoint Remote;				// endpoint receiving data from
	public int receivedDataLength;			// length of last received data
	public byte[] data = new byte[1024];
	
	// initialization
	void Awake(){
		socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
		IPEndPoint ipLocal = new IPEndPoint ( IPAddress.Any , 10001);
		socket.Bind( ipLocal );
		IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);
		Remote = (EndPoint)(sender);
		print("socket opened");
	}
	
	// Update is called once per frame
	void Update () {
		data = new byte[1024];
		if(socket.Available > 0){
			// ReceiveFrom blocks when there is no data available
			receivedDataLength = socket.ReceiveFrom(data, ref Remote);
			if(receivedDataLength > 0){
				// encode data to ASCII string for text parsing
				print(Encoding.ASCII.GetString(data, 0, receivedDataLength));
			}
		}
	}
}
