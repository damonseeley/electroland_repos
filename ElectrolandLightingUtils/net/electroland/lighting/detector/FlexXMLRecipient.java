package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.net.UnknownHostException;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.apache.log4j.Logger;

import net.electroland.broadcast.server.XMLSocketBroadcaster;
import net.electroland.broadcast.server.XMLSocketMessage;
import net.electroland.util.Util;

public class FlexXMLRecipient extends Recipient {

	private static Logger logger = Logger.getLogger(FlexXMLRecipient.class);
	private XMLSocketBroadcaster xmlsb;
	private ConcurrentLinkedQueue <String>q;

	public FlexXMLRecipient(String id, String ipStr, int port, int channels, 
							Dimension preferredDimensions) throws UnknownHostException 
	{
		super(id, ipStr, port, channels, preferredDimensions);
		q = new ConcurrentLinkedQueue<String>();
        xmlsb = new XMLSocketBroadcaster(port);
		xmlsb.start();
	}
	public FlexXMLRecipient(String id, String ipStr, int port, int channels, 
						Dimension preferredDimensions, String patchgroup) throws UnknownHostException
	{
		super(id, ipStr, port, channels, preferredDimensions, patchgroup);
		q = new ConcurrentLinkedQueue<String>();
        xmlsb = new XMLSocketBroadcaster(port);
		xmlsb.start();
	}

	public void addMessage(String message)
	{
		// we should do some < and > escaping here!
		q.add(message);
	}

	@Override
	void send(byte[] data) {
		/*
		<xml>
			<lights>
				<light id=1>255</light>
				<light id=2>128</light>
			</lights>
			<messages>
				<message id=1>play movie, gameover</message>
			</messages>
		</xml>
		 */
		StringBuffer sb = new StringBuffer("<xml>");

		// lights
		sb.append("<lights>");
		for (int i = 0; i < data.length; i++)
		{
			sb.append("<light id=\"").append(i+1).append("\">");
			sb.append(Util.unsignedByteToInt(data[i]));
			sb.append("</light>");
		}
		sb.append("</lights>");

		int count = 1;
		// messages
		sb.append("<messages>");
		while (q.size() > 0)
		{
			sb.append("<message id=\"").append(count++).append("\">");
			sb.append(q.remove());
			sb.append("</message>");
		}
		sb.append("</messages>");

		sb.append("</xml>");

		String message = sb.toString();

		logger.debug("sending flex xml message: " + message);
		xmlsb.send(new FlexRecipientXMLSocketMessage(message));
	}
}
class FlexRecipientXMLSocketMessage implements XMLSocketMessage
{
	private String message;

	public FlexRecipientXMLSocketMessage(String message){
		this.message = message;
	}
	public String toXML(){
		return message;
	}
}