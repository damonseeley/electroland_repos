#pragma strict

var connectToIP : String = "127.0.0.1";
var connectPort : int = 25001;

function OnGUI (){
		
	if (Network.peerType == NetworkPeerType.Disconnected){
	//We are currently disconnected: Not a client or host
		GUILayout.Label("Connection status: Disconnected");
		
		connectToIP = GUILayout.TextField(connectToIP, GUILayout.MinWidth(100));
		connectPort = parseInt(GUILayout.TextField(connectPort.ToString()));
		
		GUILayout.BeginVertical();
		if (GUILayout.Button ("Connect as client")){
			//Connect to the "connectToIP" and "connectPort" as entered via the GUI
			//Ignore the NAT for now
			Network.useNat = false;
			Network.Connect(connectToIP, connectPort);
		}
		
		if (GUILayout.Button ("Start Server")){
			//Start a server for 32 clients using the "connectPort" given via the GUI
			//Ignore the nat for now	
			Network.useNat = false;
			Network.InitializeServer(32, connectPort);
		}
		GUILayout.EndVertical();
		
		
	}else{
		//We've got a connection(s)!
		var offset = Camera.main.aspect * Camera.main.orthographicSize;

		if (Network.peerType == NetworkPeerType.Connecting){
		
			GUILayout.Label("Connection status: Connecting");
			
		} else if (Network.peerType == NetworkPeerType.Client){
			
			GUILayout.Label("Connection status: Client!");
			GUILayout.Label("Ping to server: "+Network.GetAveragePing(  Network.connections[0] ) );
			Camera.main.transform.position = Vector3(offset, 2, -40);	// move camera just right of center
			GUILayout.Label("Aspect Ratio: "+Camera.main.aspect);
			GUILayout.Label("Orthographic Size: "+Camera.main.orthographicSize);			
			
		} else if (Network.peerType == NetworkPeerType.Server){
			
			GUILayout.Label("Connection status: Server!");
			GUILayout.Label("Connections: "+Network.connections.length);
			Camera.main.transform.position = Vector3(-offset, 2, -40);	// move camera just left of center
			GUILayout.Label("Aspect Ratio: "+Camera.main.aspect);	
			GUILayout.Label("Orthographic Size: "+Camera.main.orthographicSize);			
			if(Network.connections.length>=1){
				GUILayout.Label("Ping to first player: "+Network.GetAveragePing(  Network.connections[0] ) );
			}			
		}

		if (GUILayout.Button ("Disconnect"))
		{
			Network.Disconnect(200);
		}
	}
	

}

// NONE of the functions below is of any use in this demo, the code below is only used for demonstration.
// First ensure you understand the code in the OnGUI() function above.

//Client functions called by Unity
function OnConnectedToServer() {
	Debug.Log("This CLIENT has connected to a server");	
}

function OnDisconnectedFromServer(info : NetworkDisconnection) {
	Debug.Log("This SERVER OR CLIENT has disconnected from a server");
}

function OnFailedToConnect(error: NetworkConnectionError){
	Debug.Log("Could not connect to server: "+ error);
}


//Server functions called by Unity
function OnPlayerConnected(player: NetworkPlayer) {
	Debug.Log("Player connected from: " + player.ipAddress +":" + player.port);
}

function OnServerInitialized() {
	Debug.Log("Server initialized and ready");
}

function OnPlayerDisconnected(player: NetworkPlayer) {
	Debug.Log("Player disconnected from: " + player.ipAddress+":" + player.port);
}


// OTHERS:
// To have a full overview of all network functions called by unity
// the next four have been added here too, but they can be ignored for now

function OnFailedToConnectToMasterServer(info: NetworkConnectionError){
	Debug.Log("Could not connect to master server: "+ info);
}

function OnNetworkInstantiate (info : NetworkMessageInfo) {
	Debug.Log("New object instantiated by " + info.sender);
}

function OnSerializeNetworkView(stream : BitStream, info : NetworkMessageInfo){
	//Custom code here (your code!)
}