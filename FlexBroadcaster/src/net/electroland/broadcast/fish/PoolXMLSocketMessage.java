package net.electroland.broadcast.fish;

import java.util.Collections;
import java.util.List;

import net.electroland.broadcast.server.XMLSocketMessage;

/**
 * Generates an XML message from a bunch of fish.
 * @author geilfuss
 *
 */
public class PoolXMLSocketMessage implements XMLSocketMessage {

	private List<Fish> fish;

	public PoolXMLSocketMessage(List<Fish> fish){
		this.fish = fish;
	}
	
	/* (non-Javadoc)
	 * @see net.electroland.fish.messaging.XMLSocketMessage#toXML()
	 */
	public String toXML() {
		StringBuffer b = new StringBuffer("<pool>"); // add sync as an attribute here.
		if (fish != null){
			Collections.sort(fish);// drop this sort if performance is an issue.
			for (Fish f : fish){
				b.append(f.toXML());
			}
		}
		b.append("</pool>");
		return b.toString();
	}
}