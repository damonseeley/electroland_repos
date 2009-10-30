package net.electroland.artnet.ip;

public class ArtNetPollReplyData extends ArtNetData {

	//	OpOutput. Transmitted low byte first. 
	short OpCode = ArtNetOpCodes.OpPollReply;
}