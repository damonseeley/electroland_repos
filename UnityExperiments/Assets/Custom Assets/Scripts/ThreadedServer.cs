using UnityEngine;
using System.Collections;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.Text;

public class ThreadedServer : MonoBehaviour {

	private Socket socket;					// socket server is binding to
	private Thread server;
	private EndPoint Remote;				// endpoint receiving data from
	private int receivedDataLength;			// length of last received data
	private byte[] data = new byte[1024];
	private bool client = false;
	
	// Use this for initialization
	void Start () {
		client = System.Convert.ToBoolean((GetComponent("SystemProperties") as SystemProperties).props.getProperty("client"));
		if(!client){
			socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
			IPEndPoint ipLocal = new IPEndPoint ( IPAddress.Any , 10001);
			socket.Bind( ipLocal );
			IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);
			Remote = (EndPoint)(sender);
			print("socket opened");
			server = new Thread(ReadData);
			server.IsBackground = true;
			server.Start();
		}
	}
	
	// Update is called once per frame
	void ReadData () {
		while(true){
			data = new byte[1024];
			while(socket.Available > 0){
				// ReceiveFrom blocks when there is no data available
				receivedDataLength = socket.ReceiveFrom(data, ref Remote);
				if(receivedDataLength > 0){
					// encode data to ASCII string for text parsing
					print(Encoding.ASCII.GetString(data, 0, receivedDataLength));
				}
			}
			// TODO: measure time it took to read data and subtract from sleep time
			Thread.Sleep(33);
		}
	}
}
