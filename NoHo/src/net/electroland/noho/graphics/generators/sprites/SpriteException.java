package net.electroland.noho.graphics.generators.sprites;

/**
 * You screwed up while managing your Sprites.  Here's why.
 * 
 * @author geilfuss
 *
 */
public class SpriteException extends Exception {

	private static final long serialVersionUID = 1L;

	public SpriteException(){
		super();
	}
	
	public SpriteException(String mssg){
		super(mssg);
	}
}
