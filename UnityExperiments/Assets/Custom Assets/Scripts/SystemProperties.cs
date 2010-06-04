using UnityEngine;
using System.Collections;

public class SystemProperties : MonoBehaviour {
	
	public string filename = "system.props";
	public Properties props;

	// Use this for initialization
	void Start () {
		props = new Properties(filename);
		Camera.main.orthographicSize = System.Convert.ToSingle(props.getProperty("cameraSize"));
		float offset = Camera.main.aspect * Camera.main.orthographicSize;
		float cameraX = System.Convert.ToSingle(props.getProperty("cameraX"));
		float cameraY = System.Convert.ToSingle(props.getProperty("cameraY"));
		float cameraZ = System.Convert.ToSingle(props.getProperty("cameraZ"));
		int cameraNumber = System.Convert.ToInt32(props.getProperty("cameraNumber"));
		//Camera.main.transform.position = new Vector3(cameraX, cameraY, cameraZ);
		if(cameraNumber == 0){				// server camera
			Camera.main.transform.position = new Vector3(cameraX, cameraY, cameraZ);
		} else if(cameraNumber == 1){	// #1 client camera
			Camera.main.transform.position = new Vector3(0 - offset, cameraY, cameraZ);
		} else if(cameraNumber == 2){	// #2 client camera
			Camera.main.transform.position = new Vector3(offset, cameraY, cameraZ);
		}
		
		float cameraRoll = System.Convert.ToSingle(props.getProperty("cameraRoll"));
		float cameraPitch = System.Convert.ToSingle(props.getProperty("cameraPitch"));
		float cameraYaw = System.Convert.ToSingle(props.getProperty("cameraYaw"));
		Camera.main.transform.Rotate(cameraPitch, cameraYaw, cameraRoll);
	}
	
	// Update is called once per frame
	void Update () {
	
	}
}
