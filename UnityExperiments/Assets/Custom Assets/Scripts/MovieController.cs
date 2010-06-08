using UnityEngine;
using System.Collections;

public class MovieController : MonoBehaviour {

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
		float HInput = Input.GetAxis("Horizontal");
		float VInput = Input.GetAxis("Vertical");
		Vector3 moveDirection = new Vector3(HInput, 0, VInput);
		float speed = 5;
		transform.Translate(speed * moveDirection * Time.deltaTime);
	}
}
