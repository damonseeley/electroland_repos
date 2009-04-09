package net.electroland.noho.graphics.generators.sprites;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Random;

import net.electroland.noho.core.NoHoConfig;
import net.electroland.noho.util.SimpleTimer;



public class LinearMotionRecSprite extends Sprite implements SpriteImageGeneratorListener {

	// x1, y1 are the start coordinates.  x2, y2 are the end coordinates
	private double x1, y1, x2, y2;
	private double dtheta = 0.025;

	// totalTime is the time that this Sprite has been given to move from
	// (x1, y1) to (x2, y2)
	private long totalTime;	
	
	// rectangle color
	private Color c;
	
	// need to have access to spriteWorld for explosion sprites
	SpriteImageGenerator spriteWorld;
	
	// "uncollide" based on a dumb timeout
	private SimpleTimer uncollide = new SimpleTimer(1);

	/**
	 * @param x1 - initial x coordinate
	 * @param y1 - initial y coordinate
	 * @param width - width of the rectangle
	 * @param height - height of the rectangle
	 * @param color - color of the rectangel
	 * @param x2 - final x coordinate
	 * @param y2 - final y coordinate
	 * @param totalTime - total time that the Sprite has been specified to move from (x1, y1) to (x2, y2).
	 * @param image - the Image to render
	 */
	public LinearMotionRecSprite(double x1, double y1, 
								double width, double height,
								Color color,
								double x2, double y2,
								long totalTime,
								SpriteImageGenerator spriteWorld){
		super(x1, y1, width, height);

		// set states
		this.x1 = x1;
		this.y1 = y1;
		this.x2 = x2;
		this.y2 = y2;
		this.totalTime = totalTime;
		this.c = color;
		this.spriteWorld = spriteWorld;
		
		setImage(new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_ARGB));

		dtheta = new Random().nextDouble()/8;
		//System.out.println(dtheta);
	}

	@Override
	public void nextFrame(long time) {
		if (time - getStartTime() >= totalTime){
			// once we've hit our destination, mark us for gc.
			kill();
		}else{
			// render it
			// LindearMotionSprite is just a square for now.
			Graphics2D g2d = getImage().createGraphics();
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f));
			g2d.setColor(Color.BLACK);
			g2d.fillRect(0, 0, getImage().getWidth(), getImage().getHeight());
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
			g2d.setColor(c);
			g2d.draw(new Rectangle(0,0,(int)getWidth() - 1, (int)getHeight() - 1));		
			g2d.draw(new Rectangle(1,1,(int)getWidth() - 3, (int)getHeight() - 3));		
			g2d.draw(new Rectangle(2,2,(int)getWidth() - 5, (int)getHeight() - 5));		

			// move it
			double percentDone = (time - getStartTime()) / (double)totalTime;
			this.setX((percentDone * (x2 - x1)) + x1);
			this.setY((percentDone * (y2 - y1)) + y1);
			this.setTheta(getTheta() + dtheta);
		}
	}
	
	public boolean isLeftToRight (){
		if (x2 > x1){
			return true;
		} else {
			return false;
		}
	}

	/* (non-Javadoc)
	 * 
	 * LinearMotionSprites are simply going to be killed off when the leave the
	 * scene.
	 * 
	 * @see net.electroland.noho.graphics.generators.sprites.SpriteAnimatorListener#exitDetected(net.electroland.noho.graphics.generators.sprites.Sprite, net.electroland.noho.graphics.generators.sprites.SpriteAnimator)
	 */
	public void exitDetected(Sprite exited, SpriteImageGenerator animator) {
		if (isDead()){
//			animator.removeListener(this);
		}else{			
			if (exited == this){
				// do nothing for now. (DO NOT CALL kill() HERE.  we may be starting
				// sprites off screen as a compensation method, so we'd be killing
				// sprites before they started
			}
		}
	}
	

	/* (non-Javadoc)
	 * 
	 * Generate a collision animation. Currently increases rotational velocity
	 * and deflects each object into different directions.
	 * 
	 * @see net.electroland.noho.graphics.generators.sprites.SpriteAnimatorListener#overlapDetected(net.electroland.noho.graphics.generators.sprites.Sprite, net.electroland.noho.graphics.generators.sprites.Sprite, net.electroland.noho.graphics.generators.sprites.SpriteAnimator)
	 */
	public void overlapDetected(Sprite one, Sprite two, SpriteImageGenerator animator) {
		if (isDead()){
//			animator.removeListener(this);
		}else{

			// only create an explosion from one sprite
			// has to be created by overlap of two LMRS's, otherwise explosion will self trigger
			if (one == this){
				if (one instanceof LinearMotionRecSprite && two instanceof LinearMotionRecSprite) {
						//try explosion sprite
						ExplosionSprite sprite;
						if (uncollide.isDone()){
//							if (this.isLeftToRight()){
							
							
							Random r = new Random();
							for (int foo = 0; foo < NoHoConfig.NUM_CONFETTI; foo++){
								
								double ex = (this.getX()+this.getWidth()/2) - (NoHoConfig.BLAST_WIDTH/2);
								double ey = (this.getY()+this.getHeight()/2) - (NoHoConfig.BLAST_HEIGHT/2);
								ex += r.nextInt(NoHoConfig.BLAST_WIDTH);
								ey += r.nextInt(NoHoConfig.BLAST_HEIGHT);
								System.out.println(ex + ", " + ey);
								
								
								sprite = new ExplosionSprite(ex,ey, 3, 3,Color.WHITE);
								spriteWorld.addSprite(sprite);
							}
							
							

								
								
								//							} else {
	//							sprite = new ExplosionSprite(this.getX()-this.getWidth()/2,this.getY(), 3, 3,Color.WHITE);
		//					}

//							spriteWorld.addSprite(sprite);

							uncollide = new SimpleTimer(1000);
						}
				}

				if (one == this || two == this){
					if (one instanceof LinearMotionRecSprite && two instanceof LinearMotionRecSprite) {
						if (((LinearMotionRecSprite)one).isLeftToRight() != ((LinearMotionRecSprite)two).isLeftToRight()) {
							//set sprites rotating
							dtheta = 10.0;
						}

					}
				}
			}
			
			
			
			//deflect one up
			if (one == this){
				//y2 = height;
			}
			// deflect the other down
			if (two == this){
				//y2 = -1.0 * height;
			}			
		}
	}
}