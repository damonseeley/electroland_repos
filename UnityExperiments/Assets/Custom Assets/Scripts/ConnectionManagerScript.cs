using UnityEngine;
using System.Collections;

// CONNECTION MANAGER
//
// Handles the instantiation and sync of game objects across the network.
// The server controls all movement of the objects, which are synched to clients using OnSerializeNetworkView.

public class ConnectionManagerScript : MonoBehaviour {

	public string connectToIP = "127.0.0.1";
	public int connectPort = 25001;
	public bool client = false;
	public Transform prefabCube;
	public Transform prefabBall;
	public Transform playerPrefab;
	
	// Use this for initialization
	void Start () {
		if (Network.peerType == NetworkPeerType.Disconnected){
			Network.useNat = false;
			client = System.Convert.ToBoolean((GetComponent("SystemProperties") as SystemProperties).props.getProperty("client"));
			if(client){
				// setup client connection
				Network.Connect(connectToIP, connectPort);
			} else {
				// setup server connection
				Network.InitializeServer(32, connectPort);
				
				// begin server based object instantiation
				Network.Instantiate(prefabCube, new Vector3(2,1.5f,0), transform.rotation, 0);
				Network.Instantiate(prefabCube, new Vector3(3,2.5f,1), transform.rotation, 0);
				Network.Instantiate(prefabCube, new Vector3(-1,3.5f,-2), transform.rotation, 0);
				Network.Instantiate(prefabBall, new Vector3(-3,2.5f,2), transform.rotation, 0);
				// this object is controlled by the user
				Network.Instantiate(playerPrefab, new Vector3(0,1.5f,0), transform.rotation, 0);
			}
		}
	}
	
	//Client functions called by Unity
	void OnConnectedToServer() {
		Debug.Log("This CLIENT has connected to a server");	
	}

	void OnDisconnectedFromServer(NetworkDisconnection info) {
		Debug.Log("This SERVER OR CLIENT has disconnected from a server");
	}

	void OnFailedToConnect(NetworkConnectionError error){
		Debug.Log("Could not connect to server: "+ error);
	}


	//Server functions called by Unity
	void OnPlayerConnected(NetworkPlayer player) {
		Debug.Log("Player connected from: " + player.ipAddress +":" + player.port);
	}

	void OnServerInitialized() {
		Debug.Log("Server initialized and ready");
	}

	void OnPlayerDisconnected(NetworkPlayer player) {
		Debug.Log("Player disconnected from: " + player.ipAddress+":" + player.port);
	}
	
	// Update is called once per frame
	void Update () {
	
	}
}
