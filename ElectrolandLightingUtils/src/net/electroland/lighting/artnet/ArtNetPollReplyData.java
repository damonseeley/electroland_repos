package net.electroland.lighting.artnet;

public class ArtNetPollReplyData extends ArtNetData {

	//	OpOutput. Transmitted low byte first. 
	short OpCode = ArtNetOpCodes.OpPollReply;
}