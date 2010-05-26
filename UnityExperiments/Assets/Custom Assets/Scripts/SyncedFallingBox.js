#pragma strict

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
}

