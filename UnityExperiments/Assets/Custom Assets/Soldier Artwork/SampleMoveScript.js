var speed = 3.0;
var rotatationSpeed = 200.0;
private var curSpeed = 0.0;

function Update () {
	// Rotate around y-axis
	var newRotation = Input.GetAxis("Horizontal") * rotatationSpeed;
	transform.Rotate(0, newRotation * Time.deltaTime, 0);

	// Calculate speed
	var newSpeed = Input.GetAxis("Vertical") * speed;
	if (Input.GetKey("left shift"))
		newSpeed *= 1.5;

	// Move the controller
	var controller : CharacterController = GetComponent (CharacterController);
	var forward = transform.TransformDirection(Vector3.forward);
	controller.SimpleMove(forward * newSpeed);
	
	// Update the speed in the Animation script
	SendMessage("SetCurrentSpeed", newSpeed, SendMessageOptions.DontRequireReceiver);
	SendMessage("SetCurrentLean", Input.GetAxis("Horizontal"), SendMessageOptions.DontRequireReceiver);
	
	// update the speed and lean across client instances not running this control script
	networkView.RPC("SetPlayerSpeed", RPCMode.All, newSpeed);
	networkView.RPC("SetPlayerLean", RPCMode.All, Input.GetAxis("Horizontal"));
	//networkView.RPC("setPlayerPosition", RPCMode.All, transform.position);
	//networkView.RPC("setPlayerRotation", RPCMode.All, transform.rotation);
}

@script RequireComponent (CharacterController)