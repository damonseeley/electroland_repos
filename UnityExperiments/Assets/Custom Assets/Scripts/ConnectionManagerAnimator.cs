using UnityEngine;
using System.Collections;

// CONNECTION MANAGER
//
// Handles the instantiation and sync of game objects across the network.
// The server controls all movement of the objects, which are synched to clients using OnSerializeNetworkView.

public class ConnectionManagerAnimator : MonoBehaviour {

	public string connectToIP = "127.0.0.1";
	public int connectPort = 25001;
	public bool client = false;
	public Transform personA;
	public Transform personB;
	public Transform personC;
	public Transform personD;
	public Transform personE;
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
				//Network.Instantiate(personA, new Vector3(2,0.5f,0), Quaternion.AngleAxis(180, Vector3.up), 0);
				//Network.Instantiate(personB, new Vector3(5,0.5f,1), Quaternion.AngleAxis(150, Vector3.up), 0);
				//Network.Instantiate(personC, new Vector3(0,0.5f,-2), Quaternion.AngleAxis(120, Vector3.up), 0);
				//Network.Instantiate(personD, new Vector3(-3,0.5f,2), Quaternion.AngleAxis(220, Vector3.up), 0);
				//Network.Instantiate(personE, new Vector3(-4,0.5f,4), Quaternion.AngleAxis(150, Vector3.up), 0);
				//Random.seed = 1;
				for(int i=0; i<2; i++){
					Network.Instantiate(personA, new Vector3((Random.value * 14) - 7, 0.5f, (Random.value * 14) - 7), Quaternion.AngleAxis(Random.value * 360, Vector3.up), 0);
					Network.Instantiate(personB, new Vector3((Random.value * 14) - 7, 0.5f, (Random.value * 14) - 7), Quaternion.AngleAxis(Random.value * 360, Vector3.up), 0);
					Network.Instantiate(personC, new Vector3((Random.value * 14) - 7, 0.5f, (Random.value * 14) - 7), Quaternion.AngleAxis(Random.value * 360, Vector3.up), 0);
					Network.Instantiate(personD, new Vector3((Random.value * 14) - 7, 0.5f, (Random.value * 14) - 7), Quaternion.AngleAxis(Random.value * 360, Vector3.up), 0);
					Network.Instantiate(personE, new Vector3((Random.value * 14) - 7, 0.5f, (Random.value * 14) - 7), Quaternion.AngleAxis(Random.value * 360, Vector3.up), 0);
				}
				// this object is controlled by the user
				Network.Instantiate(playerPrefab, new Vector3(0,0.6f,0), Quaternion.AngleAxis(180, Vector3.up), 0);
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
		// this object is controlled by the user
		//Network.Instantiate(playerPrefab, new Vector3(0,1.5f,0), transform.rotation, 0);
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
