package net.electroland.skate.core;


public class SoundNode {

	public int nodeID;
	public int soundChannel;
	public String soundFile;
	public float amplitude;
	
	public SoundNode(int id, int ch, String file, float amp )
	{
		nodeID = id;
		soundChannel = ch;
		soundFile = file;
		amplitude = amp;
	}

	
	public static void main(String args[]){

	}
}