package net.electroland.broadcast.server;

/**
 * An XMLSocketMessage is just any message that contains a method for 
 * returning some XML.
 * 
 * @author geilfuss
 */
public interface XMLSocketMessage {
	abstract public String toXML();
}