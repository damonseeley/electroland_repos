package net.electroland.EIA.core;


public class SoundNode {

	public int nodeID;
	public int soundChannel;
	public String soundFile;
	public float amplitude;
	public boolean globalSound;
	
	public SoundNode(int id, int ch, String file, float amp, boolean glSound)
	{
		nodeID = id;
		soundChannel = ch;
		soundFile = file;
		amplitude = amp;
		globalSound = glSound;
	}

	
	public static void main(String args[]){

	}
}