package net.electroland.noho.core;

/**
 * TextBundle is used to encapsolate text with its id
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class TextBundle {
	private String text;
	private int id;
	public TextBundle(int id, String text) {
		this.text = text;
		this.id = id;
	}
	
	public String getText() {
		return text;
	}
	
	public int getID() {
		return id;
	}

}
