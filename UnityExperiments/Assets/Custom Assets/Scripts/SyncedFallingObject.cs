using UnityEngine;
using System.Collections;

public class SyncedFallingObject : MonoBehaviour {

	void OnSerializeNetworkView (BitStream stream, NetworkMessageInfo info) {
		if (stream.isWriting){
			Vector3 pos = transform.position;		
			stream.Serialize(ref pos);
		} else {
			Vector3 posReceive = Vector3.zero;
			stream.Serialize(ref posReceive);
			transform.position = posReceive;
		}
	}
	
}
