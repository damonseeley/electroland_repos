#pragma strict

public var owner : NetworkPlayer;

//Last input value, we're saving this to save network messages/bandwidth.
private var lastClientHInput : float=0;
private var lastClientVInput : float=0;

//The input values the server will execute on this object
private var serverCurrentHInput : float = 0;
private var serverCurrentVInput : float = 0;

function Awake(){
	if(Network.isClient){
		enabled=false;	
	}
}

@RPC
function SetPlayer(player : NetworkPlayer){
	owner = player;
	if(player==Network.player){
		enabled=true;
	}
}

@RPC
function SendMovementInput(HInput : float, VInput : float){	
	serverCurrentHInput = HInput;
	serverCurrentVInput = VInput;
}

function OnSerializeNetworkView(stream : BitStream, info : NetworkMessageInfo){
	if (stream.isWriting){
		var pos : Vector3 = transform.position;		
		stream.Serialize(pos);
	} else {
		var posReceive : Vector3 = Vector3.zero;
		stream.Serialize(posReceive);
		transform.position = posReceive;
	}
}

function Update () {
	//Client code
	if(owner!=null && Network.player==owner){
		//Only the client that owns this object executes this code
		var HInput : float = Input.GetAxis("Horizontal");
		var VInput : float = Input.GetAxis("Vertical");
		
		//Is our input different? Do we need to update the server?
		if(lastClientHInput!=HInput || lastClientVInput!=VInput ){
			lastClientHInput = HInput;
			lastClientVInput = VInput;			
			
			if(Network.isServer){
				//Too bad a server can't send an rpc to itself using "RPCMode.Server"!...bugged :[
				SendMovementInput(HInput, VInput);
			}else if(Network.isClient){
				//SendMovementInput(HInput, VInput); //Use this (and line 64) for simple "prediction"
				networkView.RPC("SendMovementInput", RPCMode.Server, HInput, VInput);
			}
			
		}
	}
	
	//Server movement code
	if(Network.isServer){//Also enable this on the client itself: "|| Network.player==owner){|"
		//Actually move the player using his/her input
		var moveDirection : Vector3 = new Vector3(serverCurrentHInput, 0, serverCurrentVInput);
		var speed : float = 5;
		transform.Translate(speed * moveDirection * Time.deltaTime);
	}
}