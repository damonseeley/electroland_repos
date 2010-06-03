using UnityEngine;
using System.Collections;

public class NewPlayerScript : MonoBehaviour {
	
	private bool client = false;

	// Use this for initialization
	void Start () {
		GameObject conductor = GameObject.Find("Conductor");
		client = System.Convert.ToBoolean((conductor.GetComponent("SystemProperties") as SystemProperties).props.getProperty("client"));
	}
	
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
	
	// Update is called once per frame
	void Update () {
		/*
		if(!client){
			float HInput = Input.GetAxis("Horizontal");
			float VInput = Input.GetAxis("Vertical");
			Vector3 moveDirection = new Vector3(HInput, 0, VInput);
			float speed = 5;
			transform.Translate(speed * moveDirection * Time.deltaTime);
		}
		*/
	}
}
