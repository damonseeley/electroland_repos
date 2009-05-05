package net.electroland.broadcast.example;

import net.electroland.broadcast.server.XMLSocketMessage;

public class TestXMLSocketMessage implements XMLSocketMessage {

	private String message;
	public TestXMLSocketMessage(String message){
		this.message = message;
	}
	public String toXML(){
		return message;
	}
}