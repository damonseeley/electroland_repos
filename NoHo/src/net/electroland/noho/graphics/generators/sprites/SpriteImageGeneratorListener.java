package net.electroland.noho.graphics.generators.sprites;

/**
 * An interface for Objects that need to know what's going on in the Sprite kit.
 * 
 * Listeners need to add themselves to SpriteAnimatorListener.  After that,
 * they'll be notified of any Sprites overlapping, or leaving the scene.
 * 
 * (note to self: you should generate a SpriteListener for Sprites to shout out 
 *  anything interesting they might have to say about their internal state)
 * 
 * @author geilfuss
 *
 */
public interface SpriteImageGeneratorListener {
	abstract public void overlapDetected(Sprite one, Sprite two, SpriteImageGenerator animator);
	abstract public void exitDetected(Sprite exited, SpriteImageGenerator animator);
//TBD	abstract public void entryDetected(Sprite entered, SpriteImageGenerator animator);
}