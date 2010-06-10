using UnityEngine;
using System.Collections;

public class NetworkedPlayerScript : MonoBehaviour {

	private bool client = false;
	private int currentAnimation = 0;
	public string[] animations = new string[3];
	public float speed = 3.0f;
	public float rotatationSpeed = 200.0f;
	
	// Use this for initialization
	void Start () {
		animations[0] = "idle";
		animations[1] = "walk";
		animations[2] = "run";
		
		GameObject conductor = GameObject.Find("Conductor");
		client = System.Convert.ToBoolean((conductor.GetComponent("SystemProperties") as SystemProperties).props.getProperty("client"));
		if(client){
			CharacterController cc = GetComponent("CharacterController") as CharacterController;
			Destroy(cc);
			Destroy(GetComponent("SampleMoveScript"));
		}
	}
	
	void OnControllerColliderHit(ControllerColliderHit hit){
		float pushPower = 2.0f;
		Rigidbody body = hit.collider.attachedRigidbody;
		if(body == null || body.isKinematic){
			return;
		} else if(hit.moveDirection.y < -0.3){
			return;
		}
		Vector3 pushDir = new Vector3(hit.moveDirection.x, 0, hit.moveDirection.z);
		body.velocity = pushDir * pushPower;
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
	void setPlayerPosition(Vector3 pos){
		transform.position = pos;
	}
	
	[RPC]
	void setPlayerRotation(Quaternion rot){
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
	
	void Update(){
		
		if(!client){
			// Rotate around y-axis
			float newRotation = Input.GetAxis("Horizontal") * rotatationSpeed;
			transform.Rotate(0, newRotation * Time.deltaTime, 0);
			
			// Calculate speed
			float newSpeed = Input.GetAxis("Vertical") * speed;
			if (Input.GetKey("left shift")){
				newSpeed *= 1.5f;
			}
			
			// Move the controller
			CharacterController cc = GetComponent("CharacterController") as CharacterController;
			Vector3 forward = transform.TransformDirection(Vector3.forward);
			cc.SimpleMove(forward * newSpeed);
			
			// Update the speed in the Animation script
			SendMessage("SetCurrentSpeed", newSpeed, SendMessageOptions.DontRequireReceiver);
			SendMessage("SetCurrentLean", Input.GetAxis("Horizontal"), SendMessageOptions.DontRequireReceiver);
			
			// update the speed and lean across client instances not running this control script
			networkView.RPC("SetPlayerSpeed", RPCMode.All, newSpeed);
			networkView.RPC("SetPlayerLean", RPCMode.All, Input.GetAxis("Horizontal"));
			networkView.RPC("setPlayerPosition", RPCMode.All, transform.position);
			networkView.RPC("setPlayerRotation", RPCMode.All, transform.rotation);
			//networkView.RPC("setAnimationState", RPCMode.All, currentAnimation);
			//networkView.RPC("setAnimationPosition", RPCMode.All, animation[animations[currentAnimation]].normalizedTime);
		}
	}
}
