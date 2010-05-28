using UnityEngine;
using System.Collections;

public class SystemProperties : MonoBehaviour {
	
	Properties props;

	// Use this for initialization
	void Start () {
		props = new Properties("test.props");
		print(props.getProperty("title"));
	}
	
	// Update is called once per frame
	void Update () {
	
	}
}
