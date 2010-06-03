using UnityEngine;
using System.Collections;

public class SystemProperties : MonoBehaviour {
	
	public string filename = "system.props";
	public Properties props;

	// Use this for initialization
	void Start () {
		props = new Properties(filename);
		Camera.main.orthographicSize = System.Convert.ToSingle(props.getProperty("cameraSize"));
		//float offset = Camera.main.aspect * Camera.main.orthographicSize;
		float cameraX = System.Convert.ToSingle(props.getProperty("cameraX"));
		float cameraY = System.Convert.ToSingle(props.getProperty("cameraY"));
		float cameraZ = System.Convert.ToSingle(props.getProperty("cameraZ"));
		Camera.main.transform.position = new Vector3(cameraX, cameraY, cameraZ);
		
		float cameraRoll = System.Convert.ToSingle(props.getProperty("cameraRoll"));
		float cameraPitch = System.Convert.ToSingle(props.getProperty("cameraPitch"));
		float cameraYaw = System.Convert.ToSingle(props.getProperty("cameraYaw"));
		Camera.main.transform.Rotate(cameraPitch, cameraYaw, cameraRoll);
	}
	
	// Update is called once per frame
	void Update () {
	
	}
}
