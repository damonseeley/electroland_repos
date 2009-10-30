package net.electroland.artnet.ip;

public class ArtNetPollData extends ArtNetData {

	//	OpOutput. Transmitted low byte first. 
	short OpCode = ArtNetOpCodes.OpPoll;
}