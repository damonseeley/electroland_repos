package net.electroland.noho.graphics.generators.sprites;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Vector;

/**
 * A pretty simple framework for a sprite.  It's an image, x, y
 * coordinates (relative to the panel they will be rendered on) and
 * a z value for render order.
 * 
 * The SpriteAnimator only renders Sprites for whom isAlive() returns
 * true.  The SpriteAnimator will call nextFrame periodically, with
 * the time of the request passed in as an argument.  The time the
 * animation was set in motion (start() ) was last called records the
 * startTime, so you can use those two times to determine where you
 * should be in your animation task.  overlapDetected will be called
 * if the SpriteAnimator thinks you overlapped with another sprite.
 * exitDetected() will be called if it thinks you left the scene.
 * 
 * @author geilfuss
 */
abstract public class Sprite implements Comparable{

	/* (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	public int compareTo(Object other) {
		return other instanceof Sprite ? this.z - ((Sprite)other).z : -1;
	}

	// rendering parameters.  note- these are doubles so that subclasses can do
	// decimal math for objects in motion at subpixel speeds.
	private double x, y, width, height, theta;

	// z index for render order.  low z index = render first (bottom)
	private int z;
	
	// the Image to render
	private BufferedImage image;

	// whether or not this Sprite is to be rendered.  *not only does 
	// render = false disable rendering, but it also disables calls to 
	// "nextFrame()", so really it is putting the Sprite in stasis.
	private boolean render = false;

	// true = ask Animator to remove from it's store of Sprites.  As long as
	// you aren't holding an external reference, this should allow it to be
	// garbage collected.  see kill().
	private boolean dead = false;
	
	// last time this Sprite was "started".
	private long startTime;

	// listeners
	private Vector<SpriteListener>listeners; // to be implemented.

	/**
	 * 
	 * @param x
	 * @param y
	 * @param width
	 * @param height
	 */
	public Sprite(double x, double y, double width, double height){
		this.setX(x);
		this.setY(y);
		this.setWidth(width);
		this.setHeight(height);
		this.listeners = new Vector<SpriteListener>();
		try {
			start(); 
		} catch (SpriteException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Implement this to be useful.
	 * 
	 * @param time
	 */
	abstract public void nextFrame(long time);

	/**
	 * Let's the SpriteAnimator know that this thing should be in motion.
	 * @throws SpriteException
	 */
	public void start() throws SpriteException{
		if (!dead){
			startTime = System.currentTimeMillis();
			render = true;			
		}else throw new SpriteException("Sprite has been killed.");
	}
	/**
	 * 
	 *
	 */
	public void stop(){
		render = false;
	}
	/**
	 * 
	 *
	 */
	public void kill(){
		stop();
		dead = true;
	}
	/**
	 * This is a primative overlap check.  Aside from the fact that it treats
	 * the entire Sprite as rectangle, it also doesn't check to see if either
	 * Sprite may have collided on any of the tweened coordinates- and therefore
	 * you may actually have objects pass through each other.
	 * 
	 * @param other
	 * @return
	 */
	public boolean isOverlapping(Sprite other){
		return this.getRectangle().intersects(other.getRectangle());
	}
	/**
	 * Sees if the Sprit has left a scene of width by height with origin (0,0).
	 * 
	 * @param width
	 * @param height
	 * @return
	 */
	public boolean isExited(int sceneWidth, int sceneHeight){
		Rectangle scene = new Rectangle(0, 0, (int)sceneWidth, (int)sceneHeight);
		return !this.getRectangle().intersects(scene);
	}

	/**
	 * @return the dead
	 */
	public boolean isDead() {
		return dead;
	}

	/**
	 * @return the render
	 */
	public boolean isRender() {
		return render;
	}

	/**
	 * @return the startTime
	 */
	public long getStartTime() {
		return startTime;
	}

	/**
	 * @return the h
	 */
	public double getHeight() {
		return height;
	}

	/**
	 * @param h the h to set
	 */
	public void setHeight(double height) {
		this.height = height;
	}

	/**
	 * @return the image
	 */
	public BufferedImage getImage() {
		return image;
	}
	
	/**
	 * @param image the image to set
	 */
	public void setImage(BufferedImage image) {
		this.image = image;
	}

	/**
	 * @return the w
	 */
	public double getWidth() {
		return width;
	}

	/**
	 * @param w the w to set
	 */
	public void setWidth(double width) {
		this.width = width;
	}

	/**
	 * @return the x
	 */
	public double getX() {
		return x;
	}

	/**
	 * @param x the x to set
	 */
	public void setX(double x) {
		this.x = x;
	}

	/**
	 * @return the y
	 */
	public double getY() {
		return y;
	}

	/**
	 * @param y the y to set
	 */
	public void setY(double y) {
		this.y = y;
	}

	/**
	 * @return the z
	 */
	public int getZ() {
		return z;
	}

	/**
	 * @param z the z to set
	 */
	public void setZ(int z) {
		this.z = z;
	}

	/**
	 * @return the angle
	 */
	public double getTheta() {
		return theta;
	}

	/**
	 * @param angle the angle to set
	 */
	public void setTheta(double theta) {
		this.theta = theta;
	}

	/**
	 * 
	 * @return
	 */
	public Rectangle getRectangle(){
		return new Rectangle((int)x, (int)y, (int)width, (int)height);
	}
}