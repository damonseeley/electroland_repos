package net.electroland.lighting.artnet;

public class ArtNetPollData extends ArtNetData {

	//	OpOutput. Transmitted low byte first. 
	short OpCode = ArtNetOpCodes.OpPoll;
}