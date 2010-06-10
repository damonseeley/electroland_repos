using UnityEngine;
using System.Collections;

public class NetworkCharacterScript : MonoBehaviour {
	
	private bool client = false;
	private int currentAnimation = 0;
	private string[] animations = new string[1];

	// Use this for initialization
	void Start () {
		animations[0] = "mixamo.com";
		
		GameObject conductor = GameObject.Find("Conductor");
		client = System.Convert.ToBoolean((conductor.GetComponent("SystemProperties") as SystemProperties).props.getProperty("client"));
	}
	
	void OnSerializeNetworkView (BitStream stream, NetworkMessageInfo info) {
		float animTime = 0;
		if (stream.isWriting){
			animTime = animation[animations[currentAnimation]].normalizedTime;
			stream.Serialize(ref animTime);
		} else {
			stream.Serialize(ref animTime);
			animation[animations[currentAnimation]].normalizedTime = animTime;
		}
	}
	
	[RPC]
	void setCharacterPosition(Vector3 pos){
		transform.position = pos;
	}
	
	[RPC]
	void setCharacterRotation(Quaternion rot){
		transform.rotation = rot;
	}
	
	[RPC]
	void setAnimationState(int animState){
		currentAnimation = animState;
	}
	
	[RPC]
	void setAnimationPosition(float normPos){
		animation[animations[currentAnimation]].normalizedTime = normPos;
	}
	
	// Update is called once per frame
	void Update () {
		if(!client){
			//networkView.RPC("setAnimationPosition", RPCMode.All, animation[animations[currentAnimation]].normalizedTime);
		}
	}
}
