package net.electroland.noho.graphics.generators.sprites;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import net.electroland.noho.graphics.ImageGenerator;
/**
 * This is a generator for objects in motion.  
 * 
 * Given a vector of Sprite objects, the job of the Animator
 * is to render each Sprite as an image, to request that each
 * Sprite advance it's state, and to tell any of those images 
 * whether or not they've come into overlap with any others.  
 * The overlap testing is dead primative: each image is simply 
 * assumed to be a rectangle, and overlap is only checked
 *  against the whole of each rectangle.  
 * 
 * If any Sprite has exited the "scene", the SpriteAnimator
 * will call exitDetected(w,h) on the exant Sprite.
 * 
 * @author geilfuss
 */
public class SpriteImageGenerator extends ImageGenerator {

	private int z = 0;
	private int width, height;
	private Vector<Sprite>sprites;
	private Vector<SpriteImageGeneratorListener>listeners;
	
	public SpriteImageGenerator(int width, int height){
		super(width, height);
		this.width = width;
		this.height = height;
		reset();
	}
	
	@Override
	public boolean isDone() {
		// TODO Auto-generated method stub
		return false;
	}

	@SuppressWarnings("unchecked")
	@Override
	protected void render(long dt, long curTime) {

		try{
		
			Vector<Sprite>copy = (Vector<Sprite>) sprites.clone();
			for (Sprite sprite : copy){
				if (sprite.isDead()){
					sprites.remove(sprite);
	 
					// really, the sprite should have the option to
					// remove itself.  
					// BUT... that might be dangerous, as we've already thrown
					// the gauntlet that many Sprites are listeners, and you
					// don't want to accumulate a bunch of dead listeners as
					// the sriptes disappear.
					if (sprite instanceof SpriteImageGeneratorListener){
						listeners.remove(sprite);
					}
				}
			}

			// we are having some thread concurrency problems, so we're only going
			// to use a copy of sprites from here on out.
			copy = (Vector<Sprite>) sprites.clone();
			
			// z index ordered rendering. (might want to add an add dirty check to
			// avoid doing this every frame if it turns out to be any kind of real inefficiency)
			Collections.sort(copy);
	
			// temp storage of exited and overlapping pairs to tell listeners.
			Vector<Sprite> exited = new Vector();
			Vector<OverlappingPair> overlapping = new Vector();
	
			Graphics2D g2d = image.createGraphics();
			clearBackground(g2d);
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
			int place = 0;
	
			for (Sprite s : copy){
	
				// render
				s.nextFrame(curTime);
				if (s.isRender()){
					s.getImage().getHeight();	s.getImage().getWidth();		
					AffineTransform at = new AffineTransform();
					// rotate around center
					at.rotate(s.getTheta(), s.getX() + (.5*s.getImage().getWidth()), s.getY() + (.5*s.getImage().getHeight()));
					// scale to fixed height and width
					at.scale(s.getWidth() / s.getImage().getWidth(), s.getHeight() / s.getImage().getHeight());
					// move to scene coordinates (top left corner)
					at.translate(s.getX(), s.getY());
					if (s.getImage() != null){						
						g2d.drawImage(s.getImage(), at, null);
					}
				}
				
				// note if this sprite exited the scene
				if (s.isExited(width, height)){
					exited.add(s);
				}
				
				if (copy.size() > 0){
					// note if this sprite is overlapping any other
					List<Sprite> subset = sprites.subList(++place, sprites.size());
					for (Sprite other: subset){
						if (s.isOverlapping(other) && s.isRender() && other.isRender()){
							overlapping.add(new OverlappingPair(s, other));
						}
					}				
				}
			}

			// similar concurrency issue...
			Vector<SpriteImageGeneratorListener>listcopy 
				= (Vector<SpriteImageGeneratorListener>) listeners.clone();
			
			// tell the listeners
			for (SpriteImageGeneratorListener l : listcopy){
				// notify listeners of exited Sprites
				for (Sprite e : exited){
					l.exitDetected(e, this);
				}
				// notify listeners of overlapping Sprites
				for (OverlappingPair p : overlapping){
					l.overlapDetected(p.one, p.two, this);
				}
			}

		}catch(java.util.ConcurrentModificationException e){
			e.printStackTrace(System.out);
			reset();
		}
	}

	@Override
	synchronized public void reset() {
		sprites = new Vector<Sprite>();
		listeners = new Vector<SpriteImageGeneratorListener>();
		z = 0;
	}

	public SpriteImageGenerator addSprite(Sprite newSprite){
		newSprite.setZ(z++);
			sprites.add(newSprite);
		return this;
	}
	
	public SpriteImageGenerator addListener(SpriteImageGeneratorListener l){
		synchronized (listeners){
			listeners.add(l);			
		}
		return this;
	}

	public void removeListener(SpriteImageGeneratorListener l){
		synchronized (listeners){
			listeners.remove(l);
		}
	}
	
	/**
	 * private class for temp storage of overlapping Sprites
	 * @author geilfuss
	 *
	 */
	private class OverlappingPair{
		Sprite one;
		Sprite two;
		public OverlappingPair(Sprite one, Sprite two){
			this.one = one;
			this.two = two;
		}
	}
}